import os, json, pickle
import torch.optim as optim
import torch.nn as nn
from tqdm.notebook import tqdm
from qiskit.providers.fake_provider import FakeLima
from scripts.from_circ_to_numpy import operations_to_features
from scripts.model import *
import torch

print('Hello world!')

dir_models = 'experiment_models/ising'

backend = FakeLima()

def load_circuits(data_dir, f_ext='.json'):
    circuits = []
    ideal_exp_vals = []
    noisy_exp_vals = []
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f_ext)]
    for data_file in tqdm(data_files[1:], leave=True):
        if f_ext == '.pk':
            for entry in pickle.load(open(data_file, 'rb')):
                circuits.append(entry['circuit'])
                ideal_exp_vals.append(entry['ideal_exp_value'])
                noisy_exp_vals.append(entry['noisy_exp_values'])
    return circuits, ideal_exp_vals, noisy_exp_vals

train_circuits, train_ideal_exp_vals, train_noisy_exp_vals_input = load_circuits('data/ising_init_from_qasm_no_readout/train/', '.pk')
print(len(train_circuits))

test_circuits, test_ideal_exp_vals, test_noisy_exp_vals_input = load_circuits('data/ising_init_from_qasm_no_readout/val/', '.pk')
print(len(test_circuits))

train_noisy_exp_vals = [x[0] for x in train_noisy_exp_vals_input]
test_noisy_exp_vals = [x[0] for x in test_noisy_exp_vals_input]

X_train, y_train = operations_to_features(train_circuits, train_ideal_exp_vals, n_qubits=5, backend=backend)
X_test, y_test = operations_to_features(test_circuits, test_ideal_exp_vals, n_qubits=5, backend=backend)

print('Training and testing data converted to PyTorch!')

n_qubits = 4
num_epochs = 2

y_training = []
for y in range(n_qubits):
    temp = []
    for i in range(len(y_train)):
        temp.append(y_train[i][y])
    y_training.append(temp)

y_training = torch.tensor(y_training)

y_testing = []
for y in range(n_qubits):
    temp = []
    for i in range(len(y_test)):
        temp.append(y_test[i][y])
    y_testing.append(temp)

y_testing = torch.tensor(y_testing)

train_noisy_exp_vals = torch.tensor(train_noisy_exp_vals)
test_noisy_exp_vals = torch.tensor(test_noisy_exp_vals)

sequence_hidden_size_list = [1, 2, 4]
sequence_num_layers_list = [1, 2, 4]
sequence_type_list = ['LSTM']
sequence_dropout_list = [0, 0.05, 0.1]
ann_hidden_layers_list = [2, 4]
ann_hidden_units_list = [32, 64]
ann_dropout_list = [0, 0.05]
noisy_first_list = [True]

model_int = 0

for i in range(len(sequence_hidden_size_list)):
    for j in range(len(sequence_num_layers_list)):
        for k in range(len(sequence_type_list)):
            for l in range(len(sequence_dropout_list)):
                for m in range(len(ann_hidden_layers_list)):
                    for n in range(len(ann_hidden_units_list)):
                        for o in range(len(ann_dropout_list)):
                            for p in range(len(noisy_first_list)):
                                model_int += 1
                                sequence_input_size = 7
                                sequence_hidden_size = sequence_hidden_size_list[i]
                                sequence_num_layers = sequence_num_layers_list[j]
                                sequence_model_type = sequence_type_list[k]
                                sequence_dropout = sequence_dropout_list[l]
                                ann_hidden_layers = ann_hidden_layers_list[m]
                                ann_hidden_units = ann_hidden_units_list[n]
                                ann_dropout = ann_dropout_list[o]
                                noisy_first = noisy_first_list[p]

                                sequence_config = { 
                                    "input_size": sequence_input_size,
                                    "hidden_size": sequence_hidden_size,
                                    "num_layers": sequence_num_layers,
                                    "model_type": sequence_model_type,
                                    "dropout": sequence_dropout
                                }

                                ann_config = {
                                    "hidden_layers": ann_hidden_layers,
                                    "hidden_units": ann_hidden_units,
                                    "dropout": ann_dropout,
                                    "noisy_first": noisy_first
                                }

                                for q in range(n_qubits):
                                    sequence_model, ann = create_models(sequence_input_size, 
                                                                        sequence_hidden_size, 
                                                                        sequence_num_layers, 
                                                                        sequence_model_type, 
                                                                        sequence_dropout, 
                                                                        ann_hidden_layers, 
                                                                        ann_hidden_units, 
                                                                        ann_dropout, 
                                                                        noisy_first=noisy_first)
                                    loss_fn = nn.MSELoss()
                                    optimizer = optim.Adam(list(ann.parameters()) + list(sequence_model.parameters()), lr=0.001)  

                                    train_losses, test_losses = train_and_test_step(sequence_model, ann, loss_fn, optimizer, X_train, train_noisy_exp_vals[:, q], y_training[q], X_test, test_noisy_exp_vals[:, q], y_testing[q], num_epochs, noisy_first=noisy_first)

                                    save_models(sequence_model=sequence_model,
                                                ann=ann,
                                                sequence_config=sequence_config,
                                                ann_config=ann_config,
                                                save_dir=f'{dir_models}/models/model_{model_int}_{q+1}')
                                
                                    results_dir = f'{dir_models}/results'
                                    if not os.path.exists(results_dir):
                                        os.makedirs(results_dir)

                                    results_filename = os.path.join(results_dir, f'results_model_{model_int}_{q+1}.json')
                                    with open(results_filename, 'w') as f:
                                        json.dump({"train_losses": train_losses, "test_losses": test_losses}, f)

                                    print(sequence_config)
                                    print(ann_config)
                                    print(train_losses)
                                    print(test_losses)
                                    print(q)
