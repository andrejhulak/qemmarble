import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
import json, os

class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type, p_dropout):
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
        elif model_type == 'Transformer':
            #print('To be added soon, initialising with LSTM for now!')
            for i in range(5):
                self.qubit_models[f'sequence_{i}'] = nn.Transformer(d_model=input_size, nhead=hidden_size, num_encoder_layers=num_layers, batch_first=True, dropout=p_dropout)
        
    def forward(self, x):
        models_output = []
        for i, model in enumerate(self.qubit_models.values()):
            if self.model_type == 'Transformer':
                model_output = model(x[i], x[i])
            else:
                model_output, _ = model(x[i])
            model_output_last = model_output[-1, :]   
            models_output.append(model_output_last) 
        
        output = torch.stack(models_output, dim=1)
        return output
    
class ANN_first(nn.Module):
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

  def forward(self, x):
    x = torch.flatten(x)
    ReLU = nn.ReLU()

    x = ReLU(self.layers['input'](x))

    for i in range(self.nLayers):
        x = ReLU(self.layers[f'hidden{i}'](x))

        x = self.layers['output'](x)

    return x
  


class ANN(nn.Module):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               hidden_layers: int,
               output_shape: int,
               p_drouput: int,
               noisy_first: bool):
    super().__init__()
    self.layers = nn.ModuleDict()
    self.nLayers = hidden_layers
    self.noisy_first = noisy_first

    self.layers['input'] = nn.Linear(input_shape,
                                     hidden_units)

    for i in range(hidden_layers):
      self.layers[f'hidden{i}'] = nn.Linear(hidden_units,
                                            hidden_units)
      
    if noisy_first == True:
        self.layers['output'] = nn.Linear(hidden_units,
                                          output_shape)
    else:
        self.layers['output'] = nn.Linear(hidden_units + 1,
                                          output_shape)
    
    self.dropout = p_drouput

  def forward(self, x, noisy_exp_val):
    x = torch.flatten(x)
    ReLU = nn.ReLU()

    x = ReLU(self.layers['input'](x))

    for i in range(self.nLayers):
        x = ReLU(self.layers[f'hidden{i}'](x))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

    if self.noisy_first == True:
        x = self.layers['output'](x)
    else:
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
                  noisy_first: bool,
                  first_ann_hidden_layers: int,
                  first_ann_p_dropout: float,
                  n_qubits = 5):

    sequence_model = SequenceModel(input_size=sequence_input_size, hidden_size=sequence_hidden_size, num_layers=sequence_num_layers, model_type=sequence_model_type, p_dropout=sequence_dropout)
    
    if noisy_first == True:
        ann = ANN(input_shape=(sequence_hidden_size * n_qubits + 1), hidden_units=ann_hidden_units, hidden_layers=ann_hiden_layers, output_shape=1, p_drouput=ann_dropout, noisy_first=noisy_first)
    else:
        ann = ANN(input_shape=(sequence_hidden_size * n_qubits), hidden_units=ann_hidden_units, hidden_layers=ann_hiden_layers, output_shape=1, p_drouput=ann_dropout, noisy_first=noisy_first)
    
    ann_first = ANN_first(input_shape=sequence_input_size, hidden_units=sequence_input_size, hidden_layers=first_ann_hidden_layers, output_shape=sequence_input_size, p_drouput=first_ann_p_dropout)

    return sequence_model, ann, ann_first

def run_models(sequence_model, ann, ann_first, x, noisy_exp_val, noisy_first):

    y = torch.zeros(x.shape)

    # go through the first ann
    for i in range(len(x)):
        for k in range(len(x[i])):
            y[i][k] = ann_first(x[i][k])

    x = y
            
    # go through rnns
    sequence_model_output = sequence_model(x)

    # flatten for ann
    sequence_model_output = torch.flatten(sequence_model_output)

    # convert noisy exp val to appropriate dtype
    noisy_exp_val = torch.tensor(np.float32(noisy_exp_val))

    # add noisy exp val to sequence_model output
    if noisy_first == True:
        sequence_model_output_noisy_exp_val = torch.cat((sequence_model_output, noisy_exp_val.reshape(1)))
    else:
        sequence_model_output_noisy_exp_val = sequence_model_output

    # go through ann
    model_output = ann(sequence_model_output_noisy_exp_val, noisy_exp_val)

    return model_output

def train_step(sequence_model, ann, ann_first, loss_fn, X, noisy_exp_vals, y, optimiser, noisy_first):
    sequence_model.train()
    ann.train()

    train_loss = 0

    for i in range(len(X)):
        y_pred = run_models(sequence_model, ann, ann_first, X[i], noisy_exp_vals[i], noisy_first)

        loss = loss_fn(y_pred, y[i].unsqueeze(dim=0))
        train_loss += loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()
    
    return train_loss/len(X)

def test_step(sequence_model, ann, ann_first, loss_fn, X, noisy_exp_vals, y, noisy_first):
    sequence_model.eval()
    ann.eval()

    test_loss = 0
    
    with torch.inference_mode():
        for i in range(len(X)):
            y_pred = run_models(sequence_model, ann, ann_first, X[i], noisy_exp_vals[i], noisy_first)

            loss = loss_fn(y_pred, y[i].unsqueeze(dim=0))

            test_loss += loss.item()
    
        return test_loss/len(X)

def train_and_test_step(sequence_model, ann, ann_first, loss_fn, optimiser, X_train, train_noisy_exp_vals, y_train, X_test, test_noisy_exp_vals, y_test, num_epochs, noisy_first, print_results=True):
    for epoch in tqdm(range(num_epochs)):

        train_loss = train_step(sequence_model=sequence_model,
                                ann=ann,
                                ann_first=ann_first,
                                loss_fn=loss_fn,
                                X=X_train,
                                noisy_exp_vals=train_noisy_exp_vals,
                                y=y_train,
                                optimiser=optimiser, 
                                noisy_first=noisy_first)
        
        test_loss = test_step(sequence_model=sequence_model,
                              ann=ann,
                              ann_first=ann_first,
                              loss_fn=loss_fn,
                              X=X_test,
                              noisy_exp_vals=test_noisy_exp_vals,
                              y=y_test,
                              noisy_first=noisy_first)

        if epoch % 1 == 0 & print_results:
            print(f"Epoch {epoch + 1}/{num_epochs}, train loss: {np.sqrt(train_loss):.4f}, test_loss: {np.sqrt(test_loss):.4f}")
            

def save_models(sequence_model, ann, first_ann, sequence_config, ann_config, first_ann_config, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    first_ann_config_path = os.path.join(save_dir, "first_ann_config.json")
    with open(first_ann_config_path, "w") as f:
        json.dump(first_ann_config, f)

    sequence_config_path = os.path.join(save_dir, "sequence_config.json")
    with open(sequence_config_path, "w") as f:
        json.dump(sequence_config, f)

    ann_config_path = os.path.join(save_dir, "ann_config.json")
    with open(ann_config_path, "w") as f:
        json.dump(ann_config, f)

    first_ann_path = os.path.join(save_dir, "first_ann.pth")
    torch.save(first_ann.state_dict(), first_ann_path)

    sequence_model_path = os.path.join(save_dir, "sequence_model.pth")
    torch.save(sequence_model.state_dict(), sequence_model_path)

    ann_path = os.path.join(save_dir, "ann.pth")
    torch.save(ann.state_dict(), ann_path)

    print("Models and configurations saved successfully.")


def load_models(model_path):
    with open(f'{model_path}/first_ann_config.json', "r") as f:
        first_ann_config = json.load(f)

    with open(f'{model_path}/sequence_config.json', "r") as f:
        sequence_config = json.load(f)

    with open(f'{model_path}/ann_config.json', "r") as f:
        ann_config = json.load(f)

    first_ann = ANN_first(input_shape=sequence_config["input_size"],
                    hidden_units=sequence_config["input_size"],
                    hidden_layers=first_ann_config["hidden_layers"],
                    output_shape=sequence_config["input_size"],
                    p_drouput=first_ann_config["dropout"])


    sequence_model = SequenceModel(input_size=sequence_config["input_size"],
                                    hidden_size=sequence_config["hidden_size"],
                                    num_layers=sequence_config["num_layers"],
                                    model_type=sequence_config["model_type"],
                                    p_dropout=sequence_config["dropout"])

    ann = ANN(input_shape=sequence_config["hidden_size"] * 5 + 1 if ann_config["noisy_first"] else sequence_config["hidden_size"] * 5,
              hidden_units=ann_config["hidden_units"],
              hidden_layers=ann_config["hidden_layers"],
              output_shape=1,
              p_drouput=ann_config["dropout"],
              noisy_first=ann_config["noisy_first"])

    # Load model parameters
    first_ann.load_state_dict(torch.load(f'{model_path}/first_ann.pth'))
    sequence_model.load_state_dict(torch.load(f'{model_path}/sequence_model.pth'))
    ann.load_state_dict(torch.load(f'{model_path}/ann.pth'))

    print("Models and configurations loaded successfully.")
    return sequence_model, ann, first_ann