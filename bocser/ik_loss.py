import tensorflow as tf
import numpy as np
from rdkit import Chem
import ringo

import math
import tempfile
import typing

T = typing.TypeVar('T')

class CyclicCollection(typing.Generic[T]):
    """This behaves like a list but enforces cyclicity
    when accessing elements by index"""

    def __init__(self, a: list[T]) -> None:
        self.a = a

    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, idx: int) -> T:
        if not self.a:
            raise IndexError("CyclicCollection is empty")
        return self.a[idx % len(self.a)]

xx = lambda t: tuple(i + 1 for i in t)

class IKLoss:
    def __init__(
        self,
        bond_lengths: list[list[float]],
        valence_angles: list[list[float]],
        bond_lengths_dicts,
        valence_angles_dicts,
    ):
        self.bond_lengths = bond_lengths_dicts
        self.valence_angles = valence_angles_dicts

        self.D_matrices = [
            tf.stack([self.D_matrix(d) for d in cycle_lengths])
            for cycle_lengths in bond_lengths
        ]

        self.V_matrices = [
            tf.stack([self.V_matrix(a) for a in cycle_angles])
            for cycle_angles in valence_angles
        ]

    @classmethod
    def from_rdkit(cls, mol, all_ring_atoms):
        mol = Chem.AddHs(mol)
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
            Chem.MolToXYZFile(mol, tmp.name)
            p = ringo.Confpool()
            p.include_from_file(tmp.name)
        conf = p[0]

        bond_lengths = []
        valence_angles = []
        bond_lengths_dicts = []
        valence_angles_dicts = []

        for ring_atoms_list in all_ring_atoms:
            ring_atoms = CyclicCollection(ring_atoms_list)

            bl = [
                conf.l(*xx(ring_atoms[i] for i in (k, k + 1)))
                for k in range(len(ring_atoms))
            ]
            va = [
                np.deg2rad(conf.v(*xx(ring_atoms[i] for i in (k - 1, k, k + 1))))
                for k in range(len(ring_atoms))
            ]

            bond_lengths.append(bl)
            valence_angles.append(va)

            bond_lengths_dicts.append({
                tuple(ring_atoms[k] for k in (i, i + 1)): v
                for i, v in enumerate(bl)
            })
            valence_angles_dicts.append({
                tuple(ring_atoms[k] for k in (i - 1, i, i + 1)): v
                for i, v in enumerate(va)
            })

        return cls(bond_lengths, valence_angles, bond_lengths_dicts, valence_angles_dicts)

    @staticmethod
    def D_matrix(d):
        return tf.constant(
            [
                [1.0, 0.0, 0.0, d],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=tf.float64,
        )

    @staticmethod
    def V_matrix(a):
        return tf.constant(
            [
                [-math.cos(a), math.sin(a), 0.0, 0.0],
                [-math.sin(a), -math.cos(a), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=tf.float64,
        )

    @staticmethod
    def T_matrix(a):
        rotation_matrix = tf.stack([
            [tf.cos(a), -tf.sin(a)],
            [tf.sin(a), tf.cos(a)],
        ])
        return tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorFullMatrix(
                tf.ones([tf.shape(a)[0], 1, 1], dtype=tf.float64)),
            tf.linalg.LinearOperatorFullMatrix(
                tf.transpose(rotation_matrix, [2, 0, 1])),
            tf.linalg.LinearOperatorFullMatrix(
                tf.ones([tf.shape(a)[0], 1, 1], dtype=tf.float64)),
        ]).to_dense()

    def __call__(self, dihedrals_list):
        total_loss = 0
        for dihedrals, V, D in zip(dihedrals_list, self.V_matrices, self.D_matrices):
            total_motions = tf.map_fn(
                fn=lambda dihedral: tf.scan(
                    fn=tf.matmul,
                    elems=tf.map_fn(
                        fn=lambda x: x[0, :, :] @ x[1, :, :] @ x[2, :, :],
                        elems=tf.stack([V, D, self.T_matrix(dihedral)], axis=1),
                    ),
                )[-1],
                elems=dihedrals,
            )

            ik_loss = tf.map_fn(
                fn=lambda total_motion: tf.reduce_sum(
                    tf.map_fn(
                        fn=lambda x: x**2,
                        elems=tf.stack([
                            total_motion[0, 3],
                            total_motion[1, 3],
                            total_motion[2, 3],
                            total_motion[0, 0] + total_motion[1, 1] + total_motion[2, 2] - 3.0,
                            total_motion[0, 1],
                            total_motion[0, 2],
                            total_motion[1, 2],
                        ]),
                    ),
                    keepdims=True,
                ),
                elems=total_motions,
            )

            total_loss = total_loss + ik_loss

        return total_loss
