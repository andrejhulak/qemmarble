from qiskit.circuit.random import random_circuit
from scripts.utils import generate_random_pauli_sum_op
from scripts.utils import create_estimator_meas_data
import pickle, random
from qiskit.providers.fake_provider import FakeLima

def create_data(n_qubits: int, 
                circuit_depth: int, 
                pauli_terms: int, 
                pauli_coeff: float = 1.0, 
                max_entries: int = 10
                ):

    circuits_list = []
    observable_list = []
    ideal_exp_vals_list = []
    noisy_exp_vals_list = []

    backend = FakeLima()

    for i in range(max_entries):
        #circ_qubits = random.randint(1, n_qubits)

        circ_qubits = n_qubits

        qc_depth = random.randint(1, circuit_depth)

        qc = random_circuit(circ_qubits, qc_depth, measure=True)

        observable = generate_random_pauli_sum_op(circ_qubits, random.randint(1, pauli_terms), pauli_coeff)

        ideal_exp_val, noisy_exp_val = create_estimator_meas_data(
            backend=backend, circuit=qc, observable=observable
        )

        circuits_list.append(qc)
        observable_list.append(observable)
        ideal_exp_vals_list.append(ideal_exp_val)
        noisy_exp_vals_list.append(noisy_exp_val)

    return circuits_list, observable_list, ideal_exp_vals_list, noisy_exp_vals_list

def save_data(train_circuits,
              train_observables, 
              train_ideal_exp_vals, 
              train_noisy_exp_vals, 
              test_circuits, 
              test_observables,
              test_ideal_exp_vals, 
              test_noisy_exp_vals, 
              file_name
              ):

    data = {
        "train_circuits": train_circuits,
        "train_observables": train_observables,
        "train_ideal_exp_vals": train_ideal_exp_vals,
        "train_noisy_exp_vals": train_noisy_exp_vals,
        "test_circuits": test_circuits,
        "test_observables": test_observables,
        "test_ideal_exp_vals": test_ideal_exp_vals,
        "test_noisy_exp_vals": test_noisy_exp_vals
    }
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def create_and_save_data(n_qubits: int,
                         circ_depth: int,
                         pauli_terms: int,
                         n_train_circs: int,
                         n_test_circs: int,
                         file_name: str
                         ):
    
    train_circuits, train_observables, train_ideal_exp_vals, train_noisy_exp_vals = create_data(n_qubits=n_qubits,
                                                                                                circuit_depth=circ_depth,
                                                                                                pauli_terms=pauli_terms,
                                                                                                max_entries=n_train_circs
                                                                                                )

    test_circuits, test_observables, test_ideal_exp_vals, test_noisy_exp_vals = create_data(n_qubits=n_qubits,
                                                                                            circuit_depth=circ_depth,
                                                                                            pauli_terms=pauli_terms,
                                                                                            max_entries=n_test_circs
                                                                                            )
    
    save_data(train_circuits, 
              train_observables,
              train_ideal_exp_vals, 
              train_noisy_exp_vals, 
              test_circuits, 
              test_observables,
              test_ideal_exp_vals, 
              test_noisy_exp_vals, 
              file_name
              )
    
def load_data(file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    train_circuits = data["train_circuits"]
    train_observables = data['train_observables']
    train_ideal_exp_vals = data["train_ideal_exp_vals"]
    train_noisy_exp_vals = data["train_noisy_exp_vals"]
    test_circuits = data["test_circuits"]
    test_observables = data['test_observables']
    test_ideal_exp_vals = data["test_ideal_exp_vals"]
    test_noisy_exp_vals = data["test_noisy_exp_vals"]
    
    return train_circuits, train_observables, train_ideal_exp_vals, train_noisy_exp_vals, test_circuits, test_observables, test_ideal_exp_vals, test_noisy_exp_vals
