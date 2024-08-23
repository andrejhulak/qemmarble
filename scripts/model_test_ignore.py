import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
import json, os
from sklearn.metrics import root_mean_squared_error
import random

# just a note for self, i wanted to make sure that it has no difference
# whether the 2 models are in the same class or not, but pytorch's so cool that it just works

class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type, p_dropout,
                 input_shape_ann, hidden_units_ann, hidden_layers_ann, output_shape_ann, p_drouput_ann):
        super(SequenceModel, self).__init__()
        self.qubit_models = nn.ModuleDict()
        self.model_type = model_type
        # this is always gonne be 5 for now on (based on FakeLima :) )
        if model_type == 'LSTM':
            for i in range(5):
                self.qubit_models[f'sequence_{i}'] = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=p_dropout, bidirectional=False)
        elif model_type == 'RNN':
            for i in range(5):
                self.qubit_models[f'sequence_{i}'] = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=p_dropout)

        self.layers = nn.ModuleDict()
        self.nLayers = hidden_layers_ann

        self.layers['input'] = nn.Linear(input_shape_ann,
                                        hidden_units_ann)

        for i in range(hidden_layers_ann):
            self.layers[f'hidden{i}'] = nn.Linear(hidden_units_ann,
                                                    hidden_units_ann)
            
        self.layers['output'] = nn.Linear(hidden_units_ann + 1,
                                        output_shape_ann)
        
        self.dropout = p_drouput_ann
        
    def forward(self, x, noisy_exp_val):
        models_output = []
        for i, model in enumerate(self.qubit_models.values()):
            model_output, _ = model(x[i])
            model_output_last = model_output[-1, :]   
            models_output.append(model_output_last) 
        
        x = torch.stack(models_output, dim=1)

        x = torch.flatten(x)

        #x = torch.flatten(x)
        ReLU = nn.ReLU()

        noisy_exp_val = torch.tensor(np.float32(noisy_exp_val))

        x = ReLU(self.layers['input'](x))

        for i in range(self.nLayers):
            x = ReLU(self.layers[f'hidden{i}'](x))
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.layers['output'](torch.concat((x, noisy_exp_val.reshape(1))))

        return x

    
def create_models(sequence_input_size: int,
                  sequence_hidden_size: int,
                  sequence_num_layers: int,
                  sequence_model_type: str,
                  sequence_dropout: float,
                  ann_hiden_layers: int, 
                  ann_hidden_units: int,
                  ann_dropout: float,
                  n_qubits = 5):

    sequence_model = SequenceModel(input_size=sequence_input_size, 
                                   hidden_size=sequence_hidden_size, 
                                   num_layers=sequence_num_layers, 
                                   model_type=sequence_model_type, 
                                   p_dropout=sequence_dropout,
                                   input_shape_ann=(sequence_hidden_size * n_qubits),
                                   hidden_layers_ann=ann_hiden_layers,
                                   hidden_units_ann=ann_hidden_units,
                                   output_shape_ann=1,
                                   p_drouput_ann=ann_dropout)
 
    return sequence_model

def run_models(sequence_model, x, noisy_exp_val):

    return sequence_model(x, noisy_exp_val)

def train_step(sequence_model, loss_fn, X, noisy_exp_vals, y, optimiser, accumulation_steps=10):
    sequence_model.train()

    indices = list(range(len(X)))
    random.shuffle(indices)

    X = [X[i] for i in indices]
    noisy_exp_vals = [noisy_exp_vals[i] for i in indices]
    y = [y[i] for i in indices]

    optimiser.zero_grad()
    train_loss = 0.0
    total_loss = 0.0

    for i in range(len(X)):
        y_pred = run_models(sequence_model, X[i], noisy_exp_vals[i])
        current_loss = loss_fn(y_pred, y[i].unsqueeze(dim=0))

        current_loss.backward()
        total_loss += current_loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(X):
            optimiser.step()
            optimiser.zero_grad() 
            train_loss += total_loss / accumulation_steps
            total_loss = 0.0 

    return train_loss / (len(X) // accumulation_steps + (1 if len(X) % accumulation_steps != 0 else 0))



def test_step(sequence_model, X, noisy_exp_vals, y):
    sequence_model.eval()

    y_pred = []
    
    with torch.inference_mode():
        for i in range(len(X)):
            y_pred.append(run_models(sequence_model, X[i], noisy_exp_vals[i]).detach().cpu().numpy())
    
    return root_mean_squared_error(y, y_pred)

def train_and_test_step(sequence_model, loss_fn, optimiser, X_train, train_noisy_exp_vals, y_train, X_test, test_noisy_exp_vals, y_test, num_epochs, print_results=True):
    train_losses = []
    test_losses = []
    
    #for epoch in tqdm(range(num_epochs)):
    for epoch in range(num_epochs):

        train_loss = train_step(sequence_model=sequence_model,
                                loss_fn=loss_fn,
                                X=X_train,
                                noisy_exp_vals=train_noisy_exp_vals,
                                y=y_train,
                                optimiser=optimiser)
        
        test_loss = test_step(sequence_model=sequence_model,
                              X=X_test,
                              noisy_exp_vals=test_noisy_exp_vals,
                              y=y_test)

        if epoch % 1 == 0 & print_results:
            print(f"Epoch {epoch + 1}/{num_epochs}, train loss (PyTorch): {np.sqrt(train_loss).item():.4f}, test_loss (scikitlearn rmse): {test_loss:.4f}")

        train_losses.append(np.sqrt(train_loss).item())
        test_losses.append(test_loss.item())

    return train_losses, test_losses