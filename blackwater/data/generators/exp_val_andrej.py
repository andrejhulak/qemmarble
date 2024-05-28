"""Exp value generator."""
import random
from dataclasses import dataclass
from typing import List, Iterator, Dict, Any

import torch
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers import BackendV1
from torch_geometric.data import Data

from blackwater.data.utils import (
    generate_random_pauli_sum_op,
    create_estimator_meas_data,
    circuit_to_graph_data_json,
    get_backend_properties_v1,
    encode_pauli_sum_op,
)


# pylint: disable=no-member
@dataclass
class CircuitGraphData:
    """CircuitGraphData."""

    nodes: List[List[float]]
    edge_index: List[List[int]]
    edge_attr: List[List[float]]


@dataclass
class ExpValueEntry:
    """ExpValueEntry."""

    circuit_graph: Dict[str, Any]
    observable: List[List[float]]
    ideal_exp_value: float
    noisy_exp_values: List[float]
    circuit_depth: int = 0

    def __repr__(self):
        return (
            f"<ExpValueEntry ("
            f"ideal: {self.ideal_exp_value}, "
            f"noisy: {self.noisy_exp_values})>"
        )

    def to_dict(self):
        """Converts to dict."""
        return {
            "circuit_graph": self.circuit_graph,
            "observable": self.observable,
            "ideal_exp_value": self.ideal_exp_value,
            "noisy_exp_values": self.noisy_exp_values,
            "circuit_depth": self.circuit_depth,
        }

    @classmethod
    def from_json(cls, dictionary: Dict[str, Any]):
        """Creates from dict."""
        return ExpValueEntry(**dictionary)

    def to_pyg_data(self):
        """Converts to PyG data."""
        key = "DAGOpNode_wire_DAGOpNode"
        g_data = self.circuit_graph

        x = torch.tensor(g_data["nodes"]["DAGOpNode"], dtype=torch.float)
        edge_index = torch.tensor(g_data["edges"][key]["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(g_data["edges"][key]["edge_attr"], dtype=torch.float)
        y = torch.tensor([[self.ideal_exp_value]], dtype=torch.float)
        # noisy = torch.tensor([[self.noisy_exp_value]], dtype=torch.float)
        observable = torch.tensor([self.observable], dtype=torch.float)
        circuit_depth = torch.tensor([[self.circuit_depth]], dtype=torch.float)

        noisy = {}
        for idx, exp_val in enumerate(self.noisy_exp_values):
            noisy[f"noisy_{idx}"] = torch.tensor([[exp_val]], dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            # noisy=noisy,
            observable=observable,
            circuit_depth=circuit_depth,
            **noisy,
        )


def exp_value_generator(
    backend: BackendV1,
    circuits,
    observables,
    ideal_exp_vals,
    noisy_exp_vals
) -> Iterator[ExpValueEntry]:
    """Generator for exp values dataset entries.

    Args:
        backend: backend to use
        n_qubits: number of qubits in circuit
        circuit_depth: circuits depth
        pauli_terms: number of pauli terms
        pauli_coeff: pauli coeff
        max_entries: max number of entries generator can yield

    Returns:

    """
    properties = get_backend_properties_v1(backend)

    for i in range(len(circuits)):
        circuit = transpile(
            circuits[i],
            backend,
            optimization_level=0,
        )
        graph_data = circuit_to_graph_data_json(
            circuit=circuit,
            properties=properties,
            use_qubit_features=True,
            use_gate_features=True,
        )
        observable = observables[i]

        ideal_exp_val, noisy_exp_val = ideal_exp_vals[i], noisy_exp_vals[i]

        yield ExpValueEntry(
            circuit_graph=graph_data,
            observable=encode_pauli_sum_op(observable),
            ideal_exp_value=ideal_exp_val,
            noisy_exp_values=[noisy_exp_val],
        )
