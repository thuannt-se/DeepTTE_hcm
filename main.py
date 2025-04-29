import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

import csv

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--epochs', type = int, default = 100)

# evaluation args
parser.add_argument('--weight_file', type = str)
parser.add_argument('--result_file', type = str)

# cnn args
parser.add_argument('--kernel_size', type = int)

# rnn args
parser.add_argument('--pooling_method', type = str)

# multi-task args
parser.add_argument('--alpha', type = float)

# log file name
parser.add_argument('--log_file', type = str)

args = parser.parse_args()

config = json.load(open('./config.json', 'r'))

mse_file_path = "./result/mse_values.csv"  
 
 # Create the directory if it doesn't exist
os.makedirs(os.path.dirname(mse_file_path), exist_ok=True)


def train(model, elogger, train_set, eval_set):
    main_start_time = time.time()
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    mse_loss = []
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print('Training on epoch {}'.format(epoch))
        for input_file in train_set:
            print('Train on file {}'.format(input_file))

            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                _, loss = model.eval_on_batch(attr, traj, config)

                # update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                average_loss = running_loss / (idx + 1.0)
                
                print('\r Progress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(data_iter), average_loss)),
                print()
                elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, average_loss))
                epoch_training_time = time.time() - epoch_start_time
                elogger.log(f"Epoch trainning time: {epoch_training_time % 60:.1f}s")
                

        # evaluate the model after each epoch
        loss = evaluate(model, elogger, eval_set, save_result = False)

        # save the weight file after each epoch
        weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
        elogger.log('Save weight file {}'.format(weight_name))
        torch.save(model.state_dict(), './saved_weights/' + weight_name)
        mse_loss.append(loss)
    trained_time = time.time() - main_start_time
    elogger.log(f"Total trainning time: {trained_time // 60:.0f}m {trained_time % 60:.1f}s")
    write_csv(mse_loss)

def write_csv(mse_values):
  with open(mse_file_path, "w", newline="") as f:
                  writer = csv.writer(f)
                  writer.writerow(["Epoch", "MSE"])  # Write header row
                  for epoch, mse in enumerate(mse_values, start=1):
                      writer.writerow([epoch, mse])

def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]

def evaluate(model, elogger, files, save_result = False):
    mse_path = []
    model.eval()
    if save_result:
        fs = open('%s' % args.result_file, 'w')

    loss_records = []
    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            if save_result: write_result(fs, pred_dict, attr)

            running_loss += loss.item()
            average_loss = running_loss / (idx + 1.0)
        
        loss_records.append(running_loss/3600) #currently, each file contains 3600 records
        print ('Evaluate on file {}, loss {}'.format(input_file, average_loss))
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, average_loss))
    
    if save_result: fs.close()
    return np.mean(loss_records)

def get_kwargs(model_class):
    model_args = [
        param.name
        for param in inspect.signature(model_class.__init__).parameters.values()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY)
    ]  # Use parameters.values() to access parameters and filter by kind
    shell_args = args._get_kwargs()  # Assuming 'args' is a properly configured object

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if arg not in model_args:  # Use 'not in' for more concise check
            kwargs.pop(arg)

    return kwargs

def run():
    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)

    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result = True)

if __name__ == '__main__':
    run()
