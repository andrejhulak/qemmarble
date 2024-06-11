import os, json
import torch.optim as optim
import torch.nn as nn
from qiskit.providers.fake_provider import FakeLima
from scripts.data_setup import load_data
from scripts.from_circ_to_numpy import operations_to_features, save_to_json, load_from_json
from scripts.model import create_models, train_and_test_step, save_models, load_models

dir_models = 'experiments/test_2'
dir_data = 'data_small_1'

n_qubits = 5 # 5 == n_qubits, default value for now because of FakeLima
train_circuits, train_observables, train_ideal_exp_vals, train_noisy_exp_vals, test_circuits, test_observables, test_ideal_exp_vals, test_noisy_exp_vals = load_data(f'data/circuits/{dir_data}')
X_train, y_train, X_test, y_test = load_from_json(f'data/features/{dir_data}')

sequence_hidden_size_list = [1, 2, 4, 8]
sequence_num_layers_list = [1, 2, 4]
sequence_type_list = ['RNN', 'LSTM']
sequence_dropout_list = [0, 0.1]
ann_hidden_layers_list = [2, 4]
ann_hidden_units_list = [32, 64]
ann_dropout_list = [0, 0.1]
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

                                sequence_model, ann = create_models(sequence_input_size, 
                                                                    sequence_hidden_size, 
                                                                    sequence_num_layers, 
                                                                    sequence_model_type, 
                                                                    sequence_dropout, 
                                                                    ann_hidden_layers, 
                                                                    ann_hidden_units, 
                                                                    ann_dropout, 
                                                                    noisy_first=noisy_first)

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

                                loss_fn = nn.MSELoss() # leave for now
                                optimizer = optim.Adam(list(ann.parameters()) + list(sequence_model.parameters()), lr=0.001)

                                num_epochs = 5
                                train_losses, test_losses = train_and_test_step(sequence_model, ann, loss_fn, optimizer, X_train, train_noisy_exp_vals, y_train, X_test, test_noisy_exp_vals, y_test, num_epochs, noisy_first=noisy_first)

                                save_models(sequence_model=sequence_model,
                                            ann=ann,
                                            sequence_config=sequence_config,
                                            ann_config=ann_config,
                                            save_dir=f'{dir_models}/models/model_{model_int}')
                                
                                results_dir = f'{dir_models}/results'
                                if not os.path.exists(results_dir):
                                    os.makedirs(results_dir)

                                results_filename = os.path.join(results_dir, f'results_model_{model_int}.json')
                                with open(results_filename, 'w') as f:
                                    json.dump({"train_losses": train_losses, "test_losses": test_losses}, f)
                                
                                f'experiments/test_1/results/results_model_{model_int}'

models_directory = f'{dir_models}/models'
results_directory = f'{dir_models}/results'

train_losses_dict = {}

for result_file in os.listdir(results_directory):
    if result_file.endswith('.json'):
        result_path = os.path.join(results_directory, result_file)
        with open(result_path, 'r') as f:
            results = json.load(f)
        train_losses_dict[result_file] = results['test_losses']

top_5_models = sorted(train_losses_dict.items(), key=lambda x: min(x[1]))[:5]

model_info_list = []

for model_file, _ in top_5_models:
    model_name = model_file.replace('results_', '').replace('.json', '')
    model_path = f'{models_directory}/{model_name}'
    sequence_model, ann = load_models(model_path)

    with open(os.path.join(results_directory, model_file), "r") as f:
        results = json.load(f)
        test_loss = min(results['test_losses'])

    with open(f'{model_path}/sequence_config.json', "r") as f:
        sequence_config = json.load(f)

    with open(f'{model_path}/ann_config.json', "r") as f:
        ann_config = json.load(f)

    model_info_list.append({
        "model_name": model_name,
        "sequence_config": sequence_config,
        "ann_config": ann_config,
        "test_loss": test_loss
    })

sorted_model_info_list = sorted(model_info_list, key=lambda x: x["test_loss"])

for model_info in sorted_model_info_list:
    print(f"Model: {model_info['model_name']}")
    print("Sequence Config:")
    print(model_info['sequence_config'])
    print("ANN Config:")
    print(model_info['ann_config'])
    print(f"Test Loss: {model_info['test_loss']}")
    print()