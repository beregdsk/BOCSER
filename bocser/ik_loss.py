import tensorflow as tf
import numpy as np
from rdkit import Chem
import ringo
import vf3py
import networkx as nx

import math
import tempfile
import typing


class IKLoss:

    def __init__(
        self,
        bond_lengths: list[float],
        valence_angles: list[float],
        bond_lengths_dict,
        valence_angles_dict,
    ) -> None:

        self.bond_lengths = bond_lengths_dict
        self.valence_angles = valence_angles_dict

        self.D_matrices = tf.stack(
            [self.D_matrix(bond_length) for bond_length in bond_lengths])

        self.V_matrices = tf.stack(
            [self.V_matrix(valence_angle) for valence_angle in valence_angles])

    @classmethod
    def from_rdkit(cls, mol: Chem.rdchem.Mol, ring_atoms_list: list[int]):
        mol = Chem.AddHs(mol)
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
            Chem.MolToXYZFile(mol, tmp.name)
            p = ringo.Confpool()
            p.include_from_file(tmp.name)
        conf = p[0]

        ring_atoms = CyclicCollection(ring_atoms_list)

        bond_lengths = [
            conf.l(*xx(ring_atoms[index] for index in (i, i + 1)))
            for i in range(len(ring_atoms))
        ]

        valence_angles = [
            np.deg2rad(
                conf.v(*xx(ring_atoms[index] for index in (i - 1, i, i + 1))))
            for i in range(len(ring_atoms))
        ]

        bond_lengths_dict = {
            tuple(ring_atoms[index] for index in (i, i + 1)): value
            for i, value in zip(range(len(ring_atoms)), bond_lengths)
        }

        valence_angles_dict = {
            tuple(ring_atoms[index] for index in (i - 1, i, i + 1)): value
            for i, value in zip(range(len(ring_atoms)), valence_angles)
        }

        # dihedral_angles = [
        #     np.deg2rad(
        #         conf.z(*xx(ring_atoms[index]
        #                    for index in (i - 1, i, i + 1, i + 2))))
        #     for i in range(len(ring_atoms))
        # ]

        return cls(bond_lengths, valence_angles, bond_lengths_dict,
                   valence_angles_dict)

    @staticmethod
    def D_matrix(d: float) -> tf.Tensor:
        """Represents coordinate frame translation along X axis"""
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
    def V_matrix(a: float) -> tf.Tensor:
        """Represents coordinate frame rotation in XY plane by (pi - a) degrees to set valence angle to a"""
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
    def T_matrix(a: tf.Tensor) -> tf.Tensor:
        """Represents coordinate frame rotation in YZ plane by -a degrees to set dihedral angle to a
        (minus sign is just due to my convention)"""
        rotation_matrix = tf.stack([
            [tf.cos(a), -tf.sin(a)],
            [tf.sin(a), tf.cos(a)],
        ])
        return tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorFullMatrix(
                tf.ones([tf.shape(a)[0], 1, 1], dtype=tf.float64)),
            tf.linalg.LinearOperatorFullMatrix(
                tf.transpose(rotation_matrix, [2, 0, 1]), ),
            tf.linalg.LinearOperatorFullMatrix(
                tf.ones([tf.shape(a)[0], 1, 1], dtype=tf.float64)),
        ]).to_dense()

    def restore_xyz(self, dihedrals: tf.Tensor) -> tf.Tensor:

        def get_translation_part(frame: tf.Tensor) -> tf.Tensor:
            return frame[:3, 3]  # shape: (3,)

        cur_frame = tf.eye(4, dtype=tf.float64)
        atom_positions = []

        for i in range(tf.shape(dihedrals)[1]):
            atom_positions.append(get_translation_part(cur_frame))
            t_mat = self.T_matrix(dihedrals[:, i])[0]
            cur_frame = tf.linalg.matmul(cur_frame, self.V_matrices[i])
            cur_frame = tf.linalg.matmul(cur_frame, self.D_matrices[i])
            cur_frame = tf.linalg.matmul(cur_frame, t_mat)

        atom_positions.append(get_translation_part(cur_frame))

        geom = tf.stack(atom_positions)  # shape: (N+1, 3)
        mean = tf.reduce_mean(geom, axis=0, keepdims=True)
        geom_centered = geom - mean

        return geom_centered

    def __call__(self, dihedrals: tf.Tensor) -> tf.Tensor:
        # ic(dihedrals.numpy().shape)

        total_motions: tf.Tensor = tf.map_fn(
            fn=lambda dihedral: tf.scan(
                fn=tf.matmul,
                elems=tf.map_fn(
                    fn=lambda x: x[0, :, :] @ x[1, :, :] @ x[2, :, :],
                    elems=tf.stack(
                        [
                            self.V_matrices, self.D_matrices,
                            self.T_matrix(dihedral)
                        ],
                        axis=1,
                    ),
                ),
            )[-1],
            elems=dihedrals,
        )

        ik_loss: tf.Tensor = tf.map_fn(
            fn=lambda total_motion: tf.reduce_sum(
                tf.map_fn(
                    fn=lambda x: x**2,
                    elems=tf.stack([
                        total_motion[0, 3],
                        total_motion[1, 3],
                        total_motion[2, 3],
                        # Trace
                        total_motion[0, 0] + total_motion[1, 1] + total_motion[
                            2, 2] - 3.0,
                        # Off-diagonal elements for better convergence
                        total_motion[0, 1],
                        total_motion[0, 2],
                        total_motion[1, 2],
                    ])),
                keepdims=True),
            elems=total_motions)

        return ik_loss


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


def get_dihedral_angles(
    mol: Chem.rdchem.Mol
) -> tuple[list[tuple[int, int, int, int]], list[int], list[int]]:

    def get_dihedral(edge, graph: nx.Graph):
        a, b = edge
        a_nb = [nb for nb in graph.neighbors(a) if nb != b]
        if len(a_nb) == 0:
            return None
        a_nb = a_nb[0]
        b_nb = [nb for nb in graph.neighbors(b) if nb != a and nb != a_nb]
        if len(b_nb) == 0:
            return None
        b_nb = b_nb[0]
        return (a_nb, a, b, b_nb)

    graph = rdmol_to_graph(mol)
    result = [get_dihedral(edge, graph) for edge in nx.bridges(graph)]
    result = [i for i in result if i is not None]
    graph.remove_edges_from(nx.bridges(graph))

    ik_loss_dihedrals_idxs = []
    ring_traversal = None
    for comp in nx.connected_components(graph):
        if len(comp) == 1:
            continue
        subg = graph.subgraph(comp)
        simple_ring, mapping = vf3py.are_isomorphic(
            nx.cycle_graph(subg.number_of_nodes()),
            subg,
            get_mapping=True,
        )
        assert simple_ring, "Only simple lone rings are supported for now"
        assert ring_traversal is None, "Only one ring is supported for now"
        ring_traversal = CyclicCollection(
            [mapping[i] for i in range(subg.number_of_nodes())])

        assert len(
            ik_loss_dihedrals_idxs) == 0, "Only one ring is supported for now"
        first_cyclic_index = len(result)
        result += [
            tuple(ring_traversal[i + step] for step in (-1, 0, 1, 2))
            for i in range(subg.number_of_nodes())
        ]
        ik_loss_dihedrals_idxs += [
            i for i in range(first_cyclic_index, len(result))
        ]

    return result, ring_traversal.a, ik_loss_dihedrals_idxs


def rdmol_to_graph(mol: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        graph.add_node(idx, symbol=symbol, charge=charge)

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        bond_order = {
            Chem.rdchem.BondType.SINGLE: 1.0,
            Chem.rdchem.BondType.DOUBLE: 2.0,
            Chem.rdchem.BondType.TRIPLE: 3.0,
            Chem.rdchem.BondType.AROMATIC: 1.5,
        }.get(bond_type, None)

        if bond_order is None:
            raise ValueError(f"Unsupported bond type: {bond_type}")

        graph.add_edge(begin, end, bondorder=bond_order)

    for node, data in graph.nodes(data=True):
        order_sum = sum(graph[node][nb_node]['bondorder']
                        for nb_node in graph.neighbors(node))
        # assert order_sum == int(order_sum), (
        #     f"Fractional valence detected for {data['symbol']}{node+1}")
        data['valence'] = order_sum

    return graph


xx = lambda t: tuple(i + 1 for i in t)
