########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : train_deep_model
#
########################################################################

import argparse
import os
import re
import silence_tensorflow.auto
from datetime import datetime

import numpy as np
import pandas as pd

from utils.train_deep_model_utils import ModelExecutioner, json_file

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from utils.config import *
from eval_deep_model import eval_deep_model
import itertools
import torch.nn.functional as F

class GlobalAveragePooling(nn.Module):
    def __init__(self, dim):
        super(GlobalAveragePooling, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)

class FlattenAndUnsqueeze(nn.Module):
    def __init__(self):
        super(FlattenAndUnsqueeze, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1).unsqueeze(1)

class SelfAttention(nn.Module):
  def __init__(self, input_dim, input_dim_):
    super(SelfAttention, self).__init__()
    self.input_dim = input_dim
    self.query = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.key = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.value = nn.Linear(input_dim, input_dim)
    self.softmax = nn.Softmax(dim=2)
   
  def forward(self, x): # x.shape (batch_size, seq_length, input_dim)
    queries = self.query(x)
    keys = self.key(x)
    values = self.value(x)

    score = torch.bmm(queries, keys.transpose(1, 2))/(self.input_dim**0.5)
    attention = self.softmax(score)
    weighted = torch.bmm(attention, values)
    return weighted

# Function to find the fc1 layer programmatically
def find_fc_layer(module):
    for m in module.children():
        if isinstance(m, nn.Linear):
            return m
        elif isinstance(m, nn.Module):
            return find_fc_layer(m)

def train_deep_model(
        data_path,
        model_name,
        split_per,
        seed,
        read_from_file,
        batch_size,
        model_parameters_file,
        epochs,
        eval_model=False,
        transfer_learning=None,
        l2_val = 0,
        lr_rate = 1
):

        # Set up
        window_size = int(re.search(r'\d+', str(args.path)).group())
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        save_runs = 'results/runs/'
        save_weights = 'results/weights/'
        inf_time = True                 # compute inference time per timeseries

        # Load the splits
        train_set, val_set, test_set = create_splits(
                data_path,
                split_per=split_per,
                seed=10,
                read_from_file=read_from_file,
        )
        # Uncomment for testing
        if epochs == 1:
                train_set, val_set, test_set = train_set[:50], val_set[:10], test_set[:10]

        # Load the data
        training_data = TimeseriesDataset(data_path, fnames=train_set, transform=True)
        val_data = TimeseriesDataset(data_path, fnames=val_set)
        test_data = TimeseriesDataset(data_path, fnames=test_set)

        # Create the data loaders
        training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        # Compute class weights to give them to the loss function
        class_weights = training_data.get_weights_subset(device)

        # Read models parameters
        model_parameters = json_file(model_parameters_file)

        # Change input size according to input
        if 'original_length' in model_parameters:
                model_parameters['original_length'] = window_size * num_dimensions
        if 'timeseries_size' in model_parameters:
                model_parameters['timeseries_size'] = window_size * num_dimensions

        # Create the model, load it on GPU and print it
        model = deep_models[model_name.lower()](**model_parameters).to(device)
        classifier_name = f"{model_parameters_file.split('/')[-1].replace('.json', '')}_{window_size}"
        if read_from_file is not None and "unsupervised" in read_from_file:
                classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"

        learning_rate = 0.00001

        if transfer_learning:
                print('Transfer learning')
                if not torch.cuda.is_available():
                        state_dict = torch.load(transfer_learning, map_location=torch.device('cpu'))
                else:
                        state_dict = torch.load(transfer_learning)
                
                model.to(device)
                state_dict = {k: v.to(device) for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                
                for param in model.parameters():
                       param.requires_grad = False

                if "convnet" in model_name.lower():
                        # Get the number of input features for the fully connected layer
                        first_linear_layer = None
                        for layer in model.fc1:
                                if isinstance(layer, nn.Linear):
                                        first_linear_layer = layer
                                        break

                        if first_linear_layer is None:
                                raise ValueError("No Linear layer found in pretrained_model.fc")

                        num_features = first_linear_layer.in_features
                        model.fc1 = nn.Sequential(
                                nn.Linear(num_features, num_features),
                                nn.ReLU(),
                                nn.Dropout(),
                                FlattenAndUnsqueeze(),
                                SelfAttention(num_features, num_features),
                                GlobalAveragePooling(dim=1),
                                nn.Linear(num_features, len(detector_names),
                                nn.LogSoftmax(dim=1))  # Assuming 12 output classes
                        )

                        model.fc1.to(device)
                elif "inception_time" in model_name.lower():                        
                        first_linear_layer = model.linear
                        num_features = first_linear_layer.in_features
                        model.linear = nn.Sequential(
                                nn.Linear(num_features, num_features),
                                nn.ReLU(),
                                nn.Dropout(),
                                FlattenAndUnsqueeze(),
                                SelfAttention(num_features, num_features),
                                GlobalAveragePooling(dim=1),
                                nn.Linear(num_features, len(detector_names),
                                nn.LogSoftmax(dim=1))  # Assuming 12 output classes
                        )
                        model.linear.to(device)
                elif "resnet" in model_name.lower():
                        first_linear_layer = model.final
                        num_features = first_linear_layer.in_features
                        model.final = nn.Sequential(
                                nn.Linear(num_features, num_features),
                                nn.ReLU(),
                                nn.Dropout(),
                                FlattenAndUnsqueeze(),
                                SelfAttention(num_features, num_features),
                                GlobalAveragePooling(dim=1),
                                nn.Linear(num_features, len(detector_names),
                                nn.LogSoftmax(dim=1))  # Assuming 12 output classes
                        )
                        model.final.to(device)
                
                learning_rate *= lr_rate

                

        # Create the executioner object
        model_execute = ModelExecutioner(
                model=model,
                model_name=classifier_name,
                device=device,
                criterion=nn.CrossEntropyLoss(weight=class_weights).to(device),
                runs_dir=save_runs,
                weights_dir=save_weights,
                learning_rate=learning_rate,
                use_scheduler=True,
                weight_decay=learning_rate*l2_val,
        )

        # Check device of torch
        #model_execute.torch_devices_info()

        # Run training procedure
        model, results = model_execute.train(
                n_epochs=epochs,
                training_loader=training_loader,
                validation_loader=validation_loader,
                verbose=True,
        )

        # Save training stats
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        df = pd.DataFrame.from_dict(results, columns=["training_stats"], orient="index")
        df.to_csv(os.path.join(save_done_training, f"{classifier_name}_{timestamp}.csv"))

        # Evaluate on test set or val set
        if eval_model:
                if read_from_file is not None and "unsupervised" in read_from_file:
                        os.path.join(path_save_results, "unsupervised")
                eval_set = test_set if len(test_set) > 0 else val_set
                eval_deep_model(
                        data_path=data_path,
                        fnames=eval_set,
                        model_name=model_name,
                        model=model,
                        path_save=path_save_results,
                )


if __name__ == "__main__":
        parser = argparse.ArgumentParser(
                prog='run_experiment',
                description='This function is made so that we can easily run configurable experiments'
        )

        parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
        parser.add_argument('-s', '--split', type=float, help='split percentage for train and val sets', default=0.7)
        parser.add_argument('-se', '--seed', type=int, default=None, help='Seed for train/val split')
        parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
        parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
        parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", required=True)
        parser.add_argument('-b', '--batch', type=int, help='batch size', default=64)
        parser.add_argument('-ep', '--epochs', type=int, help='number of epochs', default=10)
        parser.add_argument('-e', '--eval-true', action="store_true", help='whether to evaluate the model on test data after training')
        parser.add_argument('-tlm', '--tl-model', type=str, help='path to trained model')

        args = parser.parse_args()

        grid_search = True

        if grid_search:
                l2 = list(range(0, 2, 1))
                l2 = [10*x for x in l2]
                batch_size = list(range(5, 9, 1))
                batch_size = [2**x for x in batch_size]
                lr = list(range(1, 8, 1))
                #lr = [100*x for x in lr]
                lr = [.001, .01, .1, 1, 50, 100]#, 300] #+ lr
                combinations = list(itertools.product(l2, batch_size, lr))

                if False:
                        l2 = list(range(0, 4, 1))
                        l2 = [10*x for x in l2]
                        batch_size = list(range(5, 9, 1))
                        batch_size = [2**x for x in batch_size]
                        lr = list(range(1, 8, 1))
                        #lr = [100*x for x in lr]
                        lr = [.001, .01, .1, 1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] #+ lr
                        new_combinations = list(itertools.product(l2, batch_size, lr))

                        combinations = [x for x in new_combinations if x not in combinations]

                for l2_val, batch_size_val, lr_val in combinations:
                        print('----------------------------------------------------------------')
                        print(f'Current Time: {datetime.now().strftime("%H:%M:%S")}; model: {args.model}; window_size: {args.path}; l2_val: {l2_val}; batch_size_val: {batch_size_val}; lr_val: {lr_val}')
                        train_deep_model(
                                data_path=args.path,
                                split_per=args.split,
                                seed=args.seed,
                                read_from_file=args.file,
                                model_name=args.model,
                                model_parameters_file=args.params,
                                batch_size=batch_size_val, #args.batch,
                                epochs=args.epochs,
                                eval_model=args.eval_true,
                                transfer_learning=args.tl_model,
                                l2_val=l2_val,
                                lr_rate=lr_val
                        )
        else:
                train_deep_model(
                        data_path=args.path,
                        split_per=args.split,
                        seed=args.seed,
                        read_from_file=args.file,
                        model_name=args.model,
                        model_parameters_file=args.params,
                        batch_size=128, #args.batch,
                        epochs=args.epochs,
                        eval_model=args.eval_true,
                        transfer_learning=args.tl_model
                )

