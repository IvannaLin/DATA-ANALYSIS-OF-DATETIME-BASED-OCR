import os
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.amp import autocast, GradScaler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Overriding a previously registered kernel")


from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test_easyocr import validation
import intel_extension_for_pytorch as ipex
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
        # print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train(opt, show_number = 2, amp=False):
    """ Setup early stopping conditions """
    max_training_time = None
    if opt.training_time and opt.training_time != '':
        try:
            max_training_time = float(opt.training_time) * 3600  # Convert hours to seconds
            print(f"Maximum training time set to {opt.training_time} hours ({max_training_time:.0f} seconds)")
        except ValueError:
            print(f"Invalid training_time value: {opt.training_time}. Using num_iter only.")
            max_training_time = None

    """ Setup TensorBoard """
    writer = SummaryWriter(log_dir=f'runs/{opt.experiment_name}')
    
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.experiment_name}/{opt.experiment_name}_log_dataset.txt', 'a', encoding="utf8")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers), prefetch_factor=512,
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model)
        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
        
        model = model.to(device) 
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if opt.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)  
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device) 
            model = ipex.optimize(model)
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = model.to(device)
    
    model.train() 
    # print("Model:")
    # print(model)
    total_params = count_parameters(model)
    writer.add_scalar('Model/Total_Parameters', total_params)
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # freeze some layers
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    writer.add_scalar('Model/Trainable_Parameters', sum(params_num))

    # setup optimizer
    if opt.optim=='adam':
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    with open(f'./saved_models/{opt.experiment_name}/log_model.txt', 'w', encoding="utf8") as model_log:
        model_log.write('------------ Model Architecture -------------\n')
        model_log.write(f'Training started at: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        model_log.write(str(model) + '\n\n')
        
        model_log.write('------------ Model Parameters -------------\n')
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                total_params += param_count
                model_log.write(f'{name:60s} | {str(list(param.shape)):20s} | {param_count:,}\n')
        model_log.write(f'\nTotal Trainable Parameters: {total_params:,}\n\n')
        
        model_log.write('------------ Optimizer -------------\n')
        model_log.write(str(optimizer) + '\n')

        if torch.xpu.is_available():
            model_log.write(f'\nXPU Memory Allocated: {torch.xpu.memory_allocated()/1024**2:.2f} MB')
            model_log.write(f'\nXPU Memory Cached: {torch.xpu.memory_reserved()/1024**2:.2f} MB')

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    last_checkpoint_time = start_time
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler(device='xpu')
    t1 = time.time()
    
    # Metrics tracking
    total_training_time = 0
    total_validation_time = 0
    total_images_processed = 0
    validation_count = 0
    
    while True:
        # Check if we should stop training
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Check time-based stopping condition
        if max_training_time and elapsed_time >= max_training_time:
            print(f"\nReached maximum training time of {opt.training_time} hours")
            print(f"Final iteration: {i} (of target {opt.num_iter})")
            break
            
        # Check iteration-based stopping condition
        if i > opt.num_iter:
            print('\nReached maximum number of iterations')
            break

        # train part
        optimizer.zero_grad(set_to_none=True)
        
        if amp:
            with autocast(device_type='xpu', enabled=True):
                image_tensors, labels = train_dataset.get_batch()
                batch_size = image_tensors.size(0)
                total_images_processed += batch_size
                
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    torch.backends.cudnn.enabled = False
                    cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    torch.backends.cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])  # align with Attention.forward
                    target = text[:, 1:]  # without [GO] Symbol
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            batch_size = image_tensors.size(0)
            total_images_processed += batch_size
            
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
            optimizer.step()
        
        loss_avg.add(cost)
        writer.add_scalar('Loss/train', cost.item(), i)
        
        # Calculate and log images per second
        if i % 2000 == 0:  # Log every 100 iterations to reduce noise
            current_time = time.time()
            elapsed_since_last = current_time - t1
            img_per_sec = (100 * batch_size) / elapsed_since_last
            writer.add_scalar('Performance/images_per_sec', img_per_sec, i)
            t1 = current_time

        # validation part
        if (i % opt.valInterval == 0) and (i!=0):
            training_time = time.time()-t1
            total_training_time += training_time
            
            print('training time: ', training_time)
            t1 = time.time()
            elapsed_time = time.time() - start_time
            
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)

                    confidence_file = f'./saved_models/{opt.experiment_name}/confidence_data.txt'
                    with open(confidence_file, 'a') as f:
                        # Write header if file is empty
                        if os.stat(confidence_file).st_size == 0:
                            f.write('iteration,groundtruth,prediction,confidence,correct\n')
                        
                        # Write data for each sample
                        for c, p, g in zip(confidence_score, preds, labels):
                            correct = int(p == g)
                            f.write(f'{i},{g},{p},{c:.4f},{correct}\n')

                model.train()
                gc.collect()

                # Calculate CER (Character Error Rate)
                total_chars = 0
                total_errors = 0
                for gt, pred in zip(labels, preds):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]
                    total_chars += len(gt)
                    total_errors += levenshtein_distance(gt, pred)
                
                cer = total_errors / total_chars if total_chars > 0 else 0
                
                # Calculate precision, recall, f-measure
                true_pos = sum(1 for gt, pred in zip(labels, preds) if gt == pred)
                precision = true_pos / len(preds) if len(preds) > 0 else 0
                recall = true_pos / len(labels) if len(labels) > 0 else 0
                f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Log metrics to TensorBoard
                writer.add_scalar('Metrics/Accuracy', current_accuracy, i)
                writer.add_scalar('Metrics/norm_ED', current_norm_ED, i)
                writer.add_scalar('Metrics/CER', cer, i)
                writer.add_scalar('Metrics/Precision', precision, i)
                writer.add_scalar('Metrics/Recall', recall, i)
                writer.add_scalar('Metrics/F-measure', f_measure, i)
                writer.add_scalar('Loss/validation', valid_loss, i)
                
                # training loss and validation loss
                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}, Training_time: {training_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}, {"CER":17s}: {cer:0.4f}'
                current_model_log += f'\n{"Precision":17s}: {precision:0.3f}, {"Recall":17s}: {recall:0.3f}, {"F-measure":17s}: {f_measure:0.3f}'

                # keep best accuracy model
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                start = random.randint(0,len(labels) - show_number )    
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                
                validation_time = time.time()-t1
                total_validation_time += validation_time
                validation_count += 1
                print('validation time: ', validation_time)
                log.write(f'Validation_time: {validation_time:0.5f}\n')
                
                # Log timing information
                writer.add_scalar('Time/training_time', training_time, i)
                writer.add_scalar('Time/validation_time', validation_time, i)
                writer.add_scalar('Time/avg_validation_time', total_validation_time/validation_count if validation_count > 0 else 0, i)
                writer.add_scalar('Time/total_training_time', total_training_time, i)
                writer.add_scalar('Time/total_validation_time', total_validation_time, i)
                writer.add_scalar('Time/images_processed', total_images_processed, i)
                
                t1 = time.time()
            
            # save model
            if i % opt.save_interval == 0:
                torch.save(
                    model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i}.pth')
    
        
        if i == opt.num_iter:
            print('end the training')
            writer.close()
            return
        
        # save final model when stopping
        if (max_training_time and elapsed_time >= max_training_time) or i >= opt.num_iter:
            final_model_path = f'./saved_models/{opt.experiment_name}/final_iter_{i}.pth'
            torch.save(model.state_dict(), final_model_path)
            print(f'Saved final model at iteration {i} to {final_model_path}')
            
            # Save best models one last time
            if current_accuracy > best_accuracy:
                torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
            if current_norm_ED > best_norm_ED:
                torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
            
            break

        i += 1
        
    writer.close()
    return

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]