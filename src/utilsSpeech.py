import torch
import torch.nn.functional as F
import numpy as np
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Network import *
from inputFuc import *


### DEFAULT Config ####
default_general_config = {
    'seed': 0, 
    'dt': 1., 
    'device': 'cpu', 
    'short_training_run': False,
    'visual_kernel': False
    }

default_data_config = {
    'n_class':35, 
    'sample_rate':16000,
    'preprocessing':'Mel',
    'n_fft': 400,
    'win_length': 400,
    'hop_length': 16,
    'n_mels': 64,
    'duation': 1000, #ms
    }

default_train_config = {
    'num_epochs': 150, 
    'learning_rate': 1e-2, 
    'batch_size': 32,
    'update_intervel': 200, #ms
    }

default_model_config = {
    'n_in': 64, 
    'n_out': 35, 
    'num_LP_layers': 3, 
    'num_Ins_layers': 1, 
    'LP_size': [90, 120, 120], 
    'Ins_size': [120, ], 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': [1, 10, 6], 
    'Tau1': [3, 6], 
    'Tau2': [2, 7],
    "answer_period":300,
    }



def train_batch(model, optimizer, x, y, answer_step, beta, update_period=10):
    n_steps = x.shape[1]
    avg_p = 0.
    one_hot_label = F.one_hot(y, num_classes=10)
    model.reset()
    #optimizer.zero_grad(set_to_none=False)
    prex = torch.zeros(x.shape[0], 1).to(model.device)
    for t in range(n_steps):
        r_out,_ = model.step(x[:, t])
        if n_steps-t <= answer_step:
            p = torch.softmax(r_out, dim=1)
            error = (one_hot_label - p)*beta[t]
            model.prop(error=error)
            model.backwards()
            if t%update_period==0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=False)
            avg_p = p + avg_p
        else:
            model.prop(learn=False)
    avg_p /= answer_step
    total_loss = -(one_hot_label * torch.log(avg_p+1e-7)).mean().item()        
            
    return total_loss

def train_batch_delay(model, optimizer, x, y, answer_step, beta):
    n_steps = x.shape[1]
    avg_p = 0.
    one_hot_label = F.one_hot(y, num_classes=10)
    model.reset()
    #optimizer.zero_grad(set_to_none=False)
    prex = torch.zeros(x.shape[0], 1).to(model.device)
    for t in range(n_steps):
        r_out,_ = model.step(x[:, t])
        if n_steps-t <= answer_step:
            p = torch.softmax(r_out, dim=1)
            error = (one_hot_label - p)*beta[t]
            model.prop(error=error)
            model.backwards()
            avg_p = p + avg_p
        else:
            model.prop(learn=False)
    optimizer.step()

    avg_p /= answer_step
    total_loss = -(one_hot_label * torch.log(avg_p+1e-7)).mean().item()        
            
    return total_loss


def test(model, data_loader, answer_step, beta):
    num_sample=len(data_loader.dataset)
    with torch.no_grad():
        for x_test, y_test in data_loader:
            n_steps = x_test.shape[-1]
            data = transform(data)
            num_sample += n
            model.reset()
            pred_p = 0
            for t in range(n_steps):
                r_out,_ = model.step(x_test[:, t])
                if n_steps-t <= answer_step:
                    p = torch.softmax(r_out, dim=1)
                    pred_p = p*beta[t] + pred_p
            prediction = torch.argmax(pred_p, dim=1)
            one_hot_label = F.one_hot(y_test, num_classes=35)
            loss += -(one_hot_label * torch.log(pred_p)).mean(dim=1).item()
            correct += (prediction==y_test).sum().item()

    loss /= num_sample
    acc_p = correct*100/num_sample

    return acc_p, loss


def test(model, epoch):
    model.eval()
    correct = 0
    

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")



labels = ['backward', 'bed', 'bird',
 'cat',
 'dog', 'down',
 'eight',
 'five', 'follow', 'forward', 'four',
 'go',
 'happy', 'house',
 'learn', 'left',
 'marvin',
 'nine', 'no',
 'off', 'on', 'one',
 'right',
 'seven', 'sheila', 'six', 'stop',
 'three', 'tree', 'two',
 'up',
 'visual',
 'wow',
 'yes',
 'zero']