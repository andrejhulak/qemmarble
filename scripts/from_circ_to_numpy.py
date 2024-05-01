from qiskit.compiler import transpile
from utils import get_backend_properties_v1
from scripts.gate_dict_script import gate_dict_method
import torch

def get_qc_operations(qc, backend):
    qc = transpile(circuits=qc, 
               backend=backend, 
               optimization_level=0)
    operations = []
    for operation in qc.data:
        operations.append(operation)
    return operations

# each word is -> gatetype, params, t1, t2, readout_error, gate_error, gate_length

def operations_to_features(circuits, true_exp_vals, n_qubits, backend):
    gate_dict = gate_dict_method()
    backend_properties = get_backend_properties_v1(backend)
    X = []
    i = 0
    for qc in circuits:
        #dag_circuit = circuit_to_dag(qc)
        circuit_instructions = get_qc_operations(qc, backend)
        # CircuitInstruction(operation=Instruction(name='rz', num_qu .... :)))))))
        features = []
        for i in range(n_qubits):
            features.append([])
        # qubit errors 
        for qubit_index in range(n_qubits):
            t1 = backend_properties['qubits_props'][qubit_index]['t1']
            t2 = backend_properties['qubits_props'][qubit_index]['t2']
            readout_error = backend_properties['qubits_props'][qubit_index]['readout_error']
            features[qubit_index].append([0, 0, t1, t2, readout_error, 0, 0])
        for circuit_instruction in circuit_instructions:
            gate_instruction = circuit_instruction.operation # the quantum gate applied to a qubit/qubits
            op_name = gate_instruction.name
            #print(op_name)
            op_params = gate_instruction.params

            if len(op_params) == 0:
                op_params = float('0')
            else:
                op_params = op_params[0]

            op_encoded = gate_dict[op_name]
            op_qubit = circuit_instruction.qubits
            op_qubit_index = op_qubit[0].index

            if op_name == 'cx':
                op_gate_quibts = circuit_instruction.qubits
                backend_op_name = f'{op_name}_{op_gate_quibts[0].index}_{op_gate_quibts[1].index}'
            else:
                backend_op_name = f'{op_name}_{op_qubit_index}'

            gate_error = backend_properties['gate_props'][backend_op_name]['gate_error']
            gate_length = backend_properties['gate_props'][backend_op_name]['gate_length']
            'sx_1'

            qubit_feature = [op_encoded, op_params, 0, 0, 0, gate_error, gate_length]
            features[op_qubit_index].append(qubit_feature)
            #print(f'{op_encoded} | {op_params} | {op_qubit}')

        max_length = max(len(sublist) for sublist in features)
        for sublist in features:
            while len(sublist) < max_length:
                sublist.append([float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0')])

        X.append(torch.tensor(features).float())

        

    y = []
    for exp_val in true_exp_vals:
        y.append(torch.tensor(exp_val).float())
        
    return X, y