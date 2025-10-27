from rdkit import Chem
from rdkit.Chem import AllChem
import trieste
import ringo
import tensorflow as tf
import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

from ik_loss import IKLoss

import logging
import tempfile
from typing import Optional


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class EnergyCalcResult:
    energy: float
    mol: Optional[Chem.Mol]
    new_dihedrals: Optional[tf.Tensor]


class MyObserver:

    def __init__(
        self,
        mol: Chem.rdchem.Mol,
        ik_loss: IKLoss,
        search_space_dihedrals: list[tuple[int, int, int, int]],
    ):
        self.start_mol = mol
        self.ik_loss = ik_loss
        self.search_space_dihedrals = search_space_dihedrals
        self.num_dofs = len(self.search_space_dihedrals)

    def extract_dofs_values(self, m: Chem.Mol):
        return tf.constant(
            [[
                -Chem.rdMolTransforms.GetDihedralRad(
                    m.GetConformer(),
                    *self.search_space_dihedrals[i],
                ) for i in range(self.num_dofs)
            ]],
            dtype=tf.float64,
        )

    def get_energy(
        self,
        coords: np.ndarray,
        get_mol=False,
        enforce_dihedrals=False,
    ) -> EnergyCalcResult:
        # Chem.MolToXYZFile(mol, 'test_cross.xyz')
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
            Chem.MolToMolFile(self.start_mol, tmp.name)
            tmp_mol = Chem.RWMol(Chem.MolFromMolFile(tmp.name, removeHs=False))

        mp = AllChem.MMFFGetMoleculeProperties(tmp_mol, mmffVariant='MMFF94')
        ff = AllChem.MMFFGetMoleculeForceField(tmp_mol, mp)
        for (a, b), value in self.ik_loss.bond_lengths.items():
            ff.MMFFAddDistanceConstraint(a, b, False, value, value, 1e4)
        for (a, b, c), value in self.ik_loss.valence_angles.items():
            ff.MMFFAddAngleConstraint(a, b, c, False, np.rad2deg(value),
                                      np.rad2deg(value), 1e4)
        if enforce_dihedrals:
            for (a, b, c, d), value in zip(self.search_space_dihedrals,
                                           coords):
                ff.MMFFAddTorsionConstraint(a, b, c, d, False,
                                            np.rad2deg(-value),
                                            np.rad2deg(-value), 1e4)
        ff.Minimize(maxIts=10000)

        mp = AllChem.MMFFGetMoleculeProperties(tmp_mol, mmffVariant='MMFF94')
        ff = AllChem.MMFFGetMoleculeForceField(tmp_mol, mp)
        energy = ff.CalcEnergy()
        logging.info(f"Energy is {energy}")
        result_object = EnergyCalcResult(
            energy=energy,
            mol=tmp_mol if get_mol else None,
            new_dihedrals=self.extract_dofs_values(tmp_mol)
            if not enforce_dihedrals else None,
        )
        return result_object

    def __call__(self,
                 x: tf.Tensor,
                 get_mol=False,
                 enforce_dihedrals=False) -> tf.Tensor:
        x_np = x.numpy()
        mols = []
        obs = []
        opt_x = []
        for pt in x_np:
            result = self.get_energy(pt,
                                     get_mol=get_mol,
                                     enforce_dihedrals=enforce_dihedrals)
            obs.append([result.energy])
            if get_mol:
                mols.append(result.mol)
            if not enforce_dihedrals:
                opt_x.append([result.new_dihedrals])
        obs = np.array(obs)

        dataset = trieste.data.Dataset(
            query_points=x
            if enforce_dihedrals else tf.concat([t[0] for t in opt_x], axis=0),
            observations=tf.constant(obs, dtype=tf.float64),
        )

        if get_mol:
            return dataset, mols
        else:
            return dataset


def optimize_rdmol(mol: Chem.Mol):
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    ff.Minimize(maxIts=10000)
    return ff.CalcEnergy()


class ProgressLogger:

    def __init__(
        self,
        observer: MyObserver,
        ik_loss: IKLoss,
        ik_loss_dihedrals_idxs: list[int],
        output_prefix="plot",
    ):
        self.observer = observer
        self.ik_loss = ik_loss
        self.ik_loss_dihedrals_idxs = ik_loss_dihedrals_idxs
        self.output_prefix = output_prefix
        self.p = ringo.Confpool()
        self.confpool_intialized = False

    def log(
        self,
        dataset: trieste.data.Dataset,
        mol: Chem.Mol,
        step: int,
    ) -> None:
        assert len(dataset.query_points) == 1
        ring_dihedrals = tf.gather(
            dataset.query_points,
            indices=self.ik_loss_dihedrals_idxs,
            axis=-1,
        )
        ikloss_value = self.ik_loss(ring_dihedrals).numpy()[0]
        energy = float(dataset.observations.numpy().item())

        self.add_conformation(mol,
                              f"#{step}. energy={energy} loss={ikloss_value}")
        fullopt_energy = optimize_rdmol(mol)
        self.add_conformation(mol, f"#{step}. energy={fullopt_energy}")
        self.p.save_xyz(f"step_{step}.xyz")

        logging.info(
            f"Step #{step}. IK loss={ikloss_value}. E={energy}, Eopt={fullopt_energy}"
        )

    def add_conformation(self, mol: Chem.Mol, comment: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
            Chem.MolToXYZFile(mol, tmp.name)
            self.p.include_from_file(tmp.name)
        self.p[len(self.p) - 1].descr = comment

        # if not self.confpool_intialized:
        #     logging.info("Generating isoms")
        #     self.p.generate_connectivity(0,
        #                                  mult=1.3,
        #                                  ignore_elements=['HCarbon'])
        #     self.p.generate_isomorphisms()
        #     logging.info("Done generating isoms")
        #     self.confpool_intialized = True
        # ic(self.p.get_rmsd_matrix(mirror_match=True, print_status=False))
