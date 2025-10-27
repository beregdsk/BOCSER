from rdkit import Chem
from rdkit.Chem import AllChem
import gpflow
import trieste
import ringo
import tensorflow as tf
import numpy as np

from ik_loss import (
    IKLoss,
    get_dihedral_angles,
)
from observer import (
    MyObserver,
    ProgressLogger,
)
from imp_var_with_ik import ImprovementVarianceWithIK

import logging
import timeit
from icecream import install

install()

logging.getLogger().setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')
formatter = logging.Formatter(
    '%(name)s:%(levelname)s:%(asctime)s: %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

if __name__ == "__main__":
    # mol = Chem.MolFromSmiles('C1CCCCCCC1')
    mol = Chem.MolFromSmiles('CC1CCCCCCCCC1')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    search_space_dihedrals, ring_atoms, ik_loss_dihedrals_idxs = get_dihedral_angles(
        mol)
    num_dofs = len(search_space_dihedrals)

    search_space = trieste.space.Box([-np.pi for _ in range(num_dofs)],
                                     [np.pi for _ in range(num_dofs)])

    ik_loss = IKLoss.from_rdkit(mol, ring_atoms)
    observer = MyObserver(mol, ik_loss, search_space_dihedrals)

    p = ringo.Confpool()
    p.atom_symbols = ['C' for i in range(len(ring_atoms) + 1)]

    initial_data = trieste.data.Dataset(
        query_points=tf.zeros((0, num_dofs), dtype=tf.float64),
        observations=tf.zeros((0, 1), dtype=tf.float64),
    )  # dataset is empty at this point
    for num_initial in range(100):
        AllChem.EmbedMolecule(mol)
        add_data, mols = observer(observer.extract_dofs_values(mol),
                                  get_mol=True,
                                  enforce_dihedrals=False)
        initial_data += add_data

        # Below are some actions for ik-loss validation
        cur_mol = mols[0]
        x = observer.extract_dofs_values(cur_mol)

        ring_dihedrals = tf.gather(
            x,
            indices=ik_loss_dihedrals_idxs,
            axis=-1,
        )
        cur_ikloss_value = ik_loss(ring_dihedrals)[0, 0]

        logging.info(
            f"Loss value = {cur_ikloss_value} {'[!!!]' if cur_ikloss_value > 1e-3 else '[OK]'}"
        )

        # # Print fixed IK parameters for new conformer
        # cur_conformer_obj = IKLoss.from_rdkit(cur_mol, ring_atoms)
        # logging.info(f"bond_lengths={cur_conformer_obj.bond_lengths}\n"
        #              f"valence_angles={cur_conformer_obj.valence_angles}")

        p.include_from_xyz(
            ik_loss.restore_xyz(ring_dihedrals).numpy(),
            f"loss={cur_ikloss_value}")
    # p.save_xyz("test.xyz")
    ic(initial_data) # type: ignore

    gpflow_model = gpflow.models.GPR(
        initial_data.astuple(),
        kernel=gpflow.kernels.Periodic(
            gpflow.kernels.RBF(variance=0.07,
                               lengthscales=0.005,
                               active_dims=[i for i in range(num_dofs)]),
            period=[2 * np.pi for _ in range(num_dofs)]),
        noise_variance=1e-5)
    gpflow.set_trainable(gpflow_model.likelihood, False)

    model = trieste.models.gpflow.GaussianProcessRegression(gpflow_model)

    ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(
        search_space=search_space,
        datasets=initial_data,
        models=model,
        acquisition_rule=trieste.acquisition.rule.EfficientGlobalOptimization(
            ImprovementVarianceWithIK(
                threshold=3.,
                ik_loss=ik_loss,
                ik_loss_idxs=ik_loss_dihedrals_idxs,
                ik_loss_weight=10.0,
            )),
    )

    logger = ProgressLogger(observer,
                            ik_loss,
                            ik_loss_dihedrals_idxs,
                            output_prefix="informed")
    for step in range(10):
        start = timeit.default_timer()
        new_point = ask_tell.ask()
        stop = timeit.default_timer()

        print(
            f"Time at step {step + 1}: {stop - start}; "
            f"Deepest minima: {tf.reduce_min(ask_tell.dataset.observations).numpy()}"
        )

        new_data, mols = observer(new_point,
                                  enforce_dihedrals=True,
                                  get_mol=True)
        ask_tell.tell(new_data)
        logger.log(new_data, mols[0], step)
