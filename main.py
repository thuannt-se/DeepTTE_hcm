# --- Installs (run this in a separate cell in Colab first) ---
# !pip install ujson -q

import os
# import json # Already imported if config loading works
import time
import utils # Assumes utils.py exists with to_var and normalize
import models # Assumes models/__init__.py and models/DeepTTE.py exist
import logger
import inspect
import datetime
import argparse
import data_loader # Assumes data_loader.py exists
import copy # Needed for QAT

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.quantization # Needed for QAT

from torch.autograd import Variable # Note: Variable is legacy

import numpy as np
import ujson as json # Use ujson if available for config loading

import csv

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type=str, required=True, choices=['train', 'test', 'qat_finetune'], help='Task to perform') # Added qat_finetune
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100, help='Epochs for standard training')

# model args (passed via get_kwargs if needed by model)
parser.add_argument('--kernel_size', type=int, default=3) # Example default
parser.add_argument('--pooling_method', type=str, default='attention') # Example default
parser.add_argument('--alpha', type=float, default=0.3) # Example default

# evaluation args
parser.add_argument('--weight_file', type=str, help='Path to model weights for testing or QAT pre-trained model')
parser.add_argument('--result_file', type=str, default='./result/results.txt', help='Path to save test results')

# QAT specific args
parser.add_argument('--qat_epochs', type=int, default=5, help='Epochs for QAT fine-tuning')
parser.add_argument('--qat_lr', type=float, default=1e-5, help='Learning rate for QAT fine-tuning')
parser.add_argument('--quantized_output_file', type=str, default='./saved_weights/model_quantized.pth', help='Path to save the final quantized model')
parser.add_argument('--pretrained_model', type=str, help='Alias for weight_file specifically for QAT pre-trained model')

# log file name
parser.add_argument('--log_file', type=str, default='experiment.log')

args = parser.parse_args()

# --- Load Configuration ---
try:
    # Use ujson for potentially faster JSON parsing
    config = json.load(open('./config.json', 'r'))
except FileNotFoundError:
    print("Error: config.json not found. Please ensure it exists in the current directory.")
    # Provide default fallback or exit
    config = {'train_set': ['train.json'], 'eval_set': ['eval.json'], 'test_set': ['test.json']} # Example fallback
    print("Warning: Using default config values.")
except Exception as e:
    print(f"Error loading config.json: {e}")
    exit()


# --- Ensure result/weights directories exist ---
mse_file_path = "./result/mse_values.csv"
os.makedirs(os.path.dirname(mse_file_path), exist_ok=True)
os.makedirs('./result/', exist_ok=True)
os.makedirs('./saved_weights/', exist_ok=True)
# Create ./data/ if it doesn't exist for the data loader
os.makedirs('./data/', exist_ok=True)


# --- Original train function (for FP32 training) ---
def train(model, elogger, train_set, eval_set):
    main_start_time = time.time()
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    elogger.log(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr = 1e-3) # Standard LR for FP32 training

    mse_loss = []
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- FP32 Training Epoch {epoch+1}/{args.epochs} ---")
        elogger.log(f'--- FP32 Training Epoch {epoch+1}/{args.epochs} ---')

        model.train() # Ensure model is in training mode
        total_epoch_loss = 0.0
        total_batches = 0

        for input_file in train_set:
            print(f'Train on file {input_file}')
            try:
                # Set num_workers=0 if multi-processing causes issues in Colab
                data_iter = data_loader.get_loader(input_file, args.batch_size) # Use get_loader
                if len(data_iter) == 0:
                     print(f"Warning: Data loader for {input_file} is empty. Skipping.")
                     continue
            except FileNotFoundError:
                 print(f"Error: Data file for {input_file} not found in ./data/. Skipping.")
                 continue
            except Exception as e:
                 print(f"Error creating data loader for {input_file}: {e}. Skipping.")
                 continue

            running_loss = 0.0
            pbar = tqdm(data_iter, desc=f"Epoch {epoch+1} File {input_file}")

            for idx, batch_data in enumerate(pbar):
                 # Unpack batch carefully
                 if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2:
                      print(f"Warning: Unexpected batch format: {type(batch_data)}. Skipping.")
                      continue
                 attr, traj = batch_data

                 # Ensure data is valid dictionary before proceeding
                 if not isinstance(attr, dict) or not isinstance(traj, dict):
                     print(f"Warning: Unexpected batch content format. Attr: {type(attr)}, Traj: {type(traj)}. Skipping.")
                     continue

                 try:
                     # Use .to(device) directly instead of legacy utils.to_var if possible
                     attr_dev = {k: v.to(device) for k, v in attr.items()}
                     traj_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in traj.items()}

                     # eval_on_batch calculates combined loss during training
                     _, loss = model.eval_on_batch(attr_dev, traj_dev, config)

                     # Check for NaN/Inf loss
                     if torch.isnan(loss) or torch.isinf(loss):
                          print(f"Warning: NaN or Inf loss detected at batch {idx}. Skipping step.")
                          continue

                     # update the model
                     optimizer.zero_grad()
                     loss.backward()
                     # Optional: Gradient Clipping
                     # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                     optimizer.step()

                     running_loss += loss.item()
                     average_loss = running_loss / (idx + 1.0)
                     pbar.set_postfix({'avg_loss': f'{average_loss:.4f}'})

                 except Exception as e:
                      print(f"\nError processing batch {idx} in {input_file}: {e}")
                      # Optionally log more details about the batch here for debugging
                      continue # Skip problematic batch


            file_avg_loss = running_loss / len(data_iter) if len(data_iter) > 0 else 0
            total_epoch_loss += running_loss
            total_batches += len(data_iter)
            elogger.log(f'Training Epoch {epoch+1}, File {input_file}, Avg Loss {file_avg_loss:.4f}')

        epoch_training_time = time.time() - epoch_start_time
        epoch_avg_loss = total_epoch_loss / total_batches if total_batches > 0 else 0
        print(f'\nEpoch {epoch+1} Average Loss: {epoch_avg_loss:.4f}, Time: {epoch_training_time // 60:.0f}m {epoch_training_time % 60:.1f}s')
        elogger.log(f'Epoch {epoch+1} Average Loss: {epoch_avg_loss:.4f}, Time: {epoch_training_time % 60:.1f}s')

        # Evaluate the model after each epoch
        eval_loss = evaluate(model, elogger, eval_set, save_result=False)
        mse_loss.append(eval_loss) # Append evaluation loss

        # Save the weight file after each epoch
        # Consider saving less frequently or based on validation performance improvement
        weight_name = '{}_epoch{}_{}'.format(args.log_file, epoch+1, datetime.datetime.now().strftime("%Y%m%d_%H%M"))
        save_path = os.path.join('./saved_weights/', weight_name + '.pth')
        elogger.log(f'Save weight file {save_path}')
        # Save model state_dict (handle DataParallel if necessary)
        if isinstance(model, torch.nn.DataParallel):
             torch.save(model.module.state_dict(), save_path)
        else:
             torch.save(model.state_dict(), save_path)

    trained_time = time.time() - main_start_time
    elogger.log(f"Total FP32 training time: {trained_time // 60:.0f}m {trained_time % 60:.1f}s")
    write_csv(mse_loss)


# --- QAT Fine-tuning Function ---
def qat_finetune(model_fp32, elogger, train_set, eval_set, config_dict, device):
    qat_start_time = time.time()
    elogger.log("\n--- Starting QAT Fine-tuning ---")
    elogger.log(str(model_fp32)) # Log original model structure
    elogger.log(f"QAT Args: Epochs={args.qat_epochs}, LR={args.qat_lr}")

    # Make a copy for QAT
    model_qat = copy.deepcopy(model_fp32)
    model_qat.train() # Set model to train mode BEFORE prepare_qat
    elogger.log("Copied model for QAT.")

    # Specify QAT configuration
    backend = "auto"
    if backend == "auto":
        backend = 'qnnpack' if torch.backends.quantized.engine == 'qnnpack' else 'fbgemm'
    elogger.log(f"Using QAT backend: {backend}")
    qconfig = torch.quantization.get_default_qat_qconfig(backend)
    model_qat.qconfig = qconfig
    elogger.log("QAT config set.")

    # --- Fusion Step (Highly Recommended) ---
    elogger.log("Skipping explicit fusion (HIGHLY recommend adding if applicable: Conv-BN-ReLU, etc.)")
    # Example: model_qat = torch.quantization.fuse_modules(model_qat, ...)

    # Prepare the model for QAT. Inserts FakeQuantize modules.
    torch.quantization.prepare_qat(model_qat, inplace=True)
    elogger.log("Model prepared for QAT.")
    model_qat.to(device) # Move prepared model to device

    # Define optimizer for fine-tuning (use specified small LR)
    optimizer = optim.Adam(model_qat.parameters(), lr=args.qat_lr)
    elogger.log(f"Optimizer created with LR: {args.qat_lr}")

    # --- Fine-tuning Loop ---
    elogger.log(f"Starting QAT fine-tuning loop for {args.qat_epochs} epochs...")
    for epoch in range(args.qat_epochs):
        epoch_start_time = time.time()
        print(f"\n--- QAT Fine-tuning Epoch {epoch+1}/{args.qat_epochs} ---")
        elogger.log(f'--- QAT Fine-tuning Epoch {epoch+1}/{args.qat_epochs} ---')

        model_qat.train() # Ensure model is in train mode each epoch
        total_epoch_loss = 0.0
        total_batches = 0

        for input_file in train_set:
            print(f'QAT Fine-tune on file {input_file}')
            try:
                # Set num_workers=0 if multi-processing causes issues in Colab
                data_iter = data_loader.get_loader(input_file, args.batch_size)
                if len(data_iter) == 0: continue
            except Exception as e:
                 print(f"Error creating QAT data loader for {input_file}: {e}. Skipping.")
                 continue

            running_loss = 0.0
            pbar = tqdm(data_iter, desc=f"QAT Epoch {epoch+1} File {input_file}")

            for idx, batch_data in enumerate(pbar):
                 # (Data unpacking and validation as in train function)
                 if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2: continue
                 attr, traj = batch_data
                 if not isinstance(attr, dict) or not isinstance(traj, dict): continue

                 try:
                     # Move data to device
                     attr_dev = {k: v.to(device) for k, v in attr.items()}
                     traj_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in traj.items()}

                     # Forward pass - FakeQuantize modules are active
                     # eval_on_batch calculates combined loss
                     _, loss = model_qat.eval_on_batch(attr_dev, traj_dev, config_dict)

                     if torch.isnan(loss) or torch.isinf(loss):
                          print(f"Warning: NaN or Inf loss detected during QAT at batch {idx}. Skipping step.")
                          continue

                     # Backward pass and optimize
                     optimizer.zero_grad()
                     loss.backward()
                     optimizer.step()

                     running_loss += loss.item()
                     average_loss = running_loss / (idx + 1.0)
                     pbar.set_postfix({'qat_loss': f'{average_loss:.4f}'})

                 except Exception as e:
                     print(f"\nError processing batch {idx} during QAT in {input_file}: {e}")
                     continue # Skip problematic batch

            file_avg_loss = running_loss / len(data_iter) if len(data_iter) > 0 else 0
            total_epoch_loss += running_loss
            total_batches += len(data_iter)
            elogger.log(f'QAT Epoch {epoch+1}, File {input_file}, Avg Loss {file_avg_loss:.4f}')

        epoch_training_time = time.time() - epoch_start_time
        epoch_avg_loss = total_epoch_loss / total_batches if total_batches > 0 else 0
        print(f'\nQAT Epoch {epoch+1} Average Loss: {epoch_avg_loss:.4f}, Time: {epoch_training_time // 60:.0f}m {epoch_training_time % 60:.1f}s')
        elogger.log(f'QAT Epoch {epoch+1} Average Loss: {epoch_avg_loss:.4f}, Time: {epoch_training_time % 60:.1f}s')

        # Optional: Evaluate QAT model (still in fake quant mode) on eval set after each epoch
        # Note: This evaluates the QAT model *before* final conversion
        # eval_loss_qat = evaluate(model_qat, elogger, eval_set, save_result=False)
        # elogger.log(f"QAT Epoch {epoch+1} Eval Loss (Fake Quant): {eval_loss_qat:.4f}")

    elogger.log("QAT fine-tuning finished.")
    qat_finetune_time = time.time() - qat_start_time
    elogger.log(f"Total QAT fine-tuning time: {qat_finetune_time // 60:.0f}m {qat_finetune_time % 60:.1f}s")

    # --- Convert to Quantized Model ---
    model_qat.eval() # Set to eval mode BEFORE conversion
    model_qat.cpu() # Conversion MUST be done on CPU
    elogger.log("Converting QAT model to quantized INT8 (on CPU)...")
    # Make sure the model architecture used for conversion is the same
    # as the one prepared (model_qat in this case)
    model_quantized = torch.quantization.convert(model_qat, inplace=False)
    elogger.log("Model converted to INT8.")
    model_quantized.eval()

    # --- Save the Quantized Model ---
    save_path = args.quantized_output_file
    elogger.log(f'Save quantized INT8 model state_dict to {save_path}')
    # It's recommended to save the state_dict
    torch.save(model_quantized.state_dict(), save_path)

    # --- Optional: Evaluate the FINAL Quantized Model ---
    elogger.log("Evaluating final INT8 model...")
    # Ensure evaluate function moves model to CPU if needed for INT8 inference
    final_eval_loss = evaluate(model_quantized, elogger, eval_set, save_result=False)
    elogger.log(f"Final Quantized Model Eval Loss: {final_eval_loss:.4f}")


# --- Original evaluate function (Ensure it handles device placement) ---
def evaluate(model, elogger, files, save_result = False):
    model.eval() # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Important: Move model to the appropriate device for evaluation
    # Quantized models typically run on CPU, FP32 might run on GPU
    # Check if model is quantized (has qconfig) - might need better check
    if hasattr(model, 'qconfig') and model.qconfig is not None:
         device = torch.device("cpu") # Force CPU for quantized model eval
         elogger.log("Running evaluation on CPU for quantized model.")
    model.to(device)
    elogger.log(f"Evaluation using device: {device}")


    if save_result:
        # Ensure result directory exists
        os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
        fs = open(args.result_file, 'w')

    loss_records = []
    total_loss = 0.0
    total_batches = 0
    print("\n--- Evaluating ---")

    with torch.no_grad(): # Disable gradients for evaluation
        for input_file in files:
            print(f'Evaluate on file {input_file}')
            try:
                data_iter = data_loader.get_loader(input_file, args.batch_size)
                if len(data_iter) == 0: continue
            except Exception as e:
                 print(f"Error creating eval data loader for {input_file}: {e}. Skipping.")
                 continue

            running_loss = 0.0
            pbar = tqdm(data_iter, desc=f"Evaluating {input_file}", leave=False)
            for idx, batch_data in enumerate(pbar):
                 # (Data unpacking and validation as in train function)
                 if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2: continue
                 attr, traj = batch_data
                 if not isinstance(attr, dict) or not isinstance(traj, dict): continue

                 try:
                     # Move data to device
                     attr_dev = {k: v.to(device) for k, v in attr.items()}
                     traj_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in traj.items()}

                     # Get predictions and loss
                     pred_dict, loss = model.eval_on_batch(attr_dev, traj_dev, config)

                     if torch.isnan(loss) or torch.isinf(loss):
                           print(f"Warning: NaN or Inf loss detected during evaluation at batch {idx}. Skipping loss calculation for batch.")
                           continue # Don't include NaN/Inf in average

                     if save_result: write_result(fs, pred_dict, attr_dev) # Pass attr_dev

                     running_loss += loss.item()
                     pbar.set_postfix({'avg_loss': f'{running_loss / (idx + 1.0):.4f}'})

                 except Exception as e:
                      print(f"\nError processing evaluation batch {idx} in {input_file}: {e}")
                      continue # Skip problematic batch

            file_avg_loss = running_loss / len(data_iter) if len(data_iter) > 0 else 0
            loss_records.append(file_avg_loss) # Store average loss for this file
            total_loss += running_loss
            total_batches += len(data_iter)
            print ('  Avg Loss {:.4f}'.format(file_avg_loss))
            elogger.log('Evaluate File {}, Avg Loss {:.4f}'.format(input_file, file_avg_loss))

    if save_result: fs.close()

    final_avg_loss = total_loss / total_batches if total_batches > 0 else 0
    print(f"--- Evaluation Finished. Overall Average Loss: {final_avg_loss:.4f} ---")
    elogger.log(f"--- Evaluation Finished. Overall Average Loss: {final_avg_loss:.4f} ---")
    # Return the overall average loss across all evaluation files/batches
    return final_avg_loss


# --- Original helper functions (write_csv, write_result, get_kwargs) ---
def write_csv(mse_values):
  try:
      with open(mse_file_path, "w", newline="") as f:
          writer = csv.writer(f)
          writer.writerow(["Epoch", "Eval_Loss"]) # Header, using Eval Loss now
          for epoch, loss_val in enumerate(mse_values, start=1):
              writer.writerow([epoch, loss_val])
  except Exception as e:
      print(f"Error writing CSV: {e}")


def write_result(fs, pred_dict, attr):
    # Ensure tensors are on CPU before converting to numpy
    pred = pred_dict['pred'].detach().cpu().numpy()
    label = pred_dict['label'].detach().cpu().numpy()
    # Assuming dateID/timeID are also tensors, move them too
    dateID_np = attr['dateID'].detach().cpu().numpy()
    timeID_np = attr['timeID'].detach().cpu().numpy()


    for i in range(pred.shape[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))
        # Use the numpy versions
        # dateID = dateID_np[i]
        # timeID = timeID_np[i]
        # (removed unused variables)


# Note: Need to ensure models.DeepTTE.Net is the correct class name
def get_kwargs(model_class):
    # Handles potential errors if signature cannot be inspected
    try:
        model_sig = inspect.signature(model_class.__init__)
        model_args = [
            param.name
            for param in model_sig.parameters.values()
            if param.name != 'self' and param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY, param.KEYWORD_ONLY)
        ]
    except ValueError: # Cannot inspect C-implemented functions/classes sometimes
        model_args = []
        print(f"Warning: Could not inspect arguments for {model_class}. Using only shell args.")


    shell_args_list = args._get_kwargs()
    kwargs = {}

    for arg_name, arg_val in shell_args_list:
         # Only include args that are expected by the model's __init__ OR are not known model args (pass extra args?)
         # A safer approach might be to ONLY pass known model args.
         # Let's pass only known model args + essential non-model args if needed.
         # For now, this logic passes all shell args that are ALSO model args.
         if arg_name in model_args:
              kwargs[arg_name] = arg_val
         # else:
              # print(f"Info: Argument '{arg_name}' from shell not found in model {model_class.__name__} args. Ignoring for model init.")


    # Ensure essential args from shell that might NOT be in __init__ signature are handled if needed
    # (e.g., pooling_method might be handled inside __init__ based on the arg value)
    # Add specific args from shell_args_list to kwargs if they are needed by internal logic but not direct params
    # Example: kwargs['pooling_method'] = args.pooling_method # Already handled if it's in model_args

    # Filter out None values if the __init__ doesn't handle them gracefully
    kwargs = {k: v for k, v in kwargs.items() if v is not None}


    return kwargs


# --- Main Execution Logic (Modified `run` function) ---
def run():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Ensure the assumed model class exists
    try:
        model_class = models.DeepTTE.Net # Make sure this path/name is correct
    except AttributeError:
        print("Error: Could not find model class 'models.DeepTTE.Net'. Check models/__init__.py and class name.")
        exit()

    # Get kwargs for model instantiation
    kwargs = get_kwargs(model_class)
    print(f"Model Arguments: {kwargs}")

    # Experiment logger
    elogger = logger.Logger(args.log_file)
    elogger.log(f"Task: {args.task}")
    elogger.log(f"Using device: {device}")
    elogger.log(f"Parsed Shell Arguments: {vars(args)}") # Log all parsed args

    # Instantiate the FP32 model structure for all tasks initially
    model_fp32 = model_class(**kwargs)

    # --- Task Handling ---
    if args.task == 'train':
        elogger.log("Starting FP32 Training...")
        train(model_fp32, elogger, train_set=config.get('train_set', []), eval_set=config.get('eval_set', []))

    elif args.task == 'test':
        elogger.log("Starting Testing...")
        if not args.weight_file or not os.path.exists(args.weight_file):
            elogger.log(f"Error: Weight file not specified or found for testing: {args.weight_file}")
            exit()
        try:
            elogger.log(f"Loading weights from: {args.weight_file}")
            state_dict = torch.load(args.weight_file, map_location='cpu')
            # Handle potential DataParallel/module prefix
            if isinstance(model_fp32, torch.nn.DataParallel):
                 model_fp32.module.load_state_dict(state_dict)
            else:
                 # Adjust keys if necessary (e.g., remove 'module.' prefix)
                 # state_dict = {k.replace('module.',''): v for k, v in state_dict.items()}
                 model_fp32.load_state_dict(state_dict)

            model_fp32.to(device) # Move model to device for evaluation
            evaluate(model_fp32, elogger, config.get('test_set', []), save_result=True)
        except Exception as e:
            elogger.log(f"Error during testing: {e}")
            exit()

    elif args.task == 'qat_finetune':
        elogger.log("Starting QAT Fine-tuning...")
        # Determine the pre-trained model path
        pretrained_path = args.pretrained_model if args.pretrained_model else args.weight_file
        if not pretrained_path or not os.path.exists(pretrained_path):
            elogger.log(f"Error: Pre-trained model path not specified or found for QAT: {pretrained_path}")
            exit()

        try:
            elogger.log(f"Loading pre-trained FP32 weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu') # Load to CPU first
            # (Handle DataParallel/module prefix when loading)
            if isinstance(model_fp32, torch.nn.DataParallel):
                 model_fp32.module.load_state_dict(state_dict)
            else:
                 model_fp32.load_state_dict(state_dict) # Adjust keys if necessary

            # Call the QAT function
            qat_finetune(model_fp32, # Pass the loaded FP32 model
                         elogger,
                         train_set=config.get('train_set', []),
                         eval_set=config.get('eval_set', []),
                         config_dict=config, # Pass the global config dict
                         device=device)
        except Exception as e:
            elogger.log(f"Error during QAT fine-tuning process: {e}")
            exit()

if __name__ == '__main__':
    # Ensure ./data directory exists before data loader tries to access it
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("Created ./data/ directory. Please place data files inside.")

    run()