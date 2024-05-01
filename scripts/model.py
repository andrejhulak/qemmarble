import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type, p_dropout):
        super(SequenceModel, self).__init__()
        self.qubit_models = nn.ModuleDict()
        # this is always gonne be 5 for now on (based on FakeLima :) )
        if model_type == 'LSTM':
            for i in range(5):
                self.qubit_models[f'sequence_{i}'] = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=p_dropout)
        elif model_type == 'RNN':
            for i in range(5):
                self.qubit_models[f'sequence_{i}'] = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=p_dropout)
        elif model_type == 'Transformer':
            print('To be added soon, initialising with LSTM for now!')
            for i in range(5):
                self.qubit_models[f'sequence_{i}'] = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=p_dropout)
        
    def forward(self, x):
        models_output = []
        for i, model in enumerate(self.qubit_models.values()):
            model_output, _ = model(x[i])
            model_output_last = model_output[-1, :]   
            models_output.append(model_output_last) 
        
        output = torch.stack(models_output, dim=1)
        return output


class ANN(nn.Module):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               hidden_layers: int,
               output_shape: int,
               p_drouput: int):
    super().__init__()
    self.layers = nn.ModuleDict()
    self.nLayers = hidden_layers

    self.layers['input'] = nn.Linear(input_shape,
                                     hidden_units)

    for i in range(hidden_layers):
      self.layers[f'hidden{i}'] = nn.Linear(hidden_units,
                                            hidden_units)

    self.layers['output'] = nn.Linear(hidden_units,
                                      output_shape)
    
    self.dropout = p_drouput

  def forward(self, x):
    x = torch.flatten(x)
    ReLU = nn.ReLU()

    x = ReLU(self.layers['input'](x))

    for i in range(self.nLayers):
        x = ReLU(self.layers[f'hidden{i}'](x))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

    x = self.layers['output'](x)

    return x
  
    
def create_models(sequence_input_size: int,
                  sequence_hidden_size: int,
                  sequence_num_layers: int,
                  sequence_model_type: str,
                  sequence_dropout: int,
                  ann_hiden_layers: int, 
                  ann_hidden_units: int,
                  ann_dropout: int,
                  n_qubits = 5):

    sequence_model = SequenceModel(input_size=sequence_input_size, hidden_size=sequence_hidden_size, num_layers=sequence_num_layers, model_type=sequence_model_type, p_dropout=sequence_dropout)
    ann = ANN(input_shape=(sequence_hidden_size * n_qubits + 1), hidden_units=ann_hidden_units, hidden_layers=ann_hiden_layers, output_shape=1, p_drouput=ann_dropout)

    return sequence_model, ann

def run_models(sequence_model, ann, x, noisy_exp_val):
    # go through rnns
    sequence_model_output = sequence_model(x)

    # flatten for ann
    sequence_model_output = torch.flatten(sequence_model_output)

    # convert noisy exp val to appropriate dtype
    noisy_exp_val = torch.tensor(np.float32(noisy_exp_val))

    # add noisy exp val to sequence_model output
    sequence_model_output_noisy_exp_val = torch.cat((sequence_model_output, noisy_exp_val.reshape(1)))

    # go through ann
    model_output = ann(sequence_model_output_noisy_exp_val)

    return model_output

def train_step(sequence_model, ann, loss_fn, X, noisy_exp_vals, y, optimiser):
    sequence_model.train()
    ann.train()

    train_loss = 0

    for i in range(len(X)):
        y_pred = run_models(sequence_model, ann, X[i], noisy_exp_vals[i])

        loss = loss_fn(y_pred, y[i].unsqueeze(dim=0))
        train_loss += loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()
    
    return train_loss/len(X)

def test_step(sequence_model, ann, loss_fn, X, noisy_exp_vals, y):
    sequence_model.eval()
    ann.eval()

    test_loss = 0
    
    with torch.inference_mode():
        for i in range(len(X)):
            y_pred = run_models(sequence_model, ann, X[i], noisy_exp_vals[i])

            loss = loss_fn(y_pred, y[i].unsqueeze(dim=0))

            test_loss += loss.item()
    
        return test_loss/len(X)

def train_and_test_step(sequence_model, ann, loss_fn, optimiser, X_train, train_noisy_exp_vals, y_train, X_test, test_noisy_exp_vals, y_test, num_epochs, print_results=True):
    for epoch in tqdm(range(num_epochs)):

        train_loss = train_step(sequence_model=sequence_model,
                                ann=ann,
                                loss_fn=loss_fn,
                                X=X_train,
                                noisy_exp_vals=train_noisy_exp_vals,
                                y=y_train,
                                optimiser=optimiser)
        
        test_loss = test_step(sequence_model=sequence_model,
                              ann=ann,
                              loss_fn=loss_fn,
                              X=X_test,
                              noisy_exp_vals=test_noisy_exp_vals,
                              y=y_test)

        if epoch % 1 == 0 & print_results:
            print(f"Epoch {epoch + 1}/{num_epochs}, train loss: {np.sqrt(train_loss):.4f}, test_loss: {np.sqrt(test_loss):.4f}")
