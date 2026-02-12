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
    'num_samples': 5000,
    'train_split': 0.8,
    'template_len': 12,
    'padding': [36,60],
    'scale_coeff': .4, 
    'max_translation': 48,
    'corr_noise_scale': 0.25,
    'iid_noise_scale': 2e-2,
    'shear_scale': 0.75,
    'shuffle_seq': False,
    'duration': 360, #ms
    'final_seq_length': 360,
    'seed': 42,
    'url': 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl',
    'prepad': 5,      
    }

default_train_config = {
    'num_epochs': 150, 
    'learning_rate': 1e-2, 
    'batch_size': 100, 
    }

default_model_config = {
    'n_in': 1, 
    'n_out': 10, 
    'num_LP_layers': 4, 
    'num_Ins_layers': 1, 
    'LP_size': [60, 90, 90, 90], 
    'Ins_size': [120, ], 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': [1, 10, 6], 
    'Tau1': [3, 6], 
    'Tau2': [2, 7],
    'Tau3': [1, 8.0, 12.0],
    "answer_period":300,
    }


def train_batch(model, optimizer, x, y, answer_step, pad_steps, beta, update_period=10):
    n_steps = x.shape[1]
    avg_p = 0.
    one_hot_label = F.one_hot(y, num_classes=10)
    model.reset()
    #optimizer.zero_grad(set_to_none=False)
    prex = torch.zeros(x.shape[0], 1).to(model.device)
    for t in range(pad_steps):
        r_out,_ = model.step(prex)
        model.prop(learn=False)
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

def train_batch_delay(model, optimizer, x, y, answer_step, pad_steps, beta):
    n_steps = x.shape[1]
    avg_p = 0.
    one_hot_label = F.one_hot(y, num_classes=10)
    model.reset()
    #optimizer.zero_grad(set_to_none=False)
    prex = torch.zeros(x.shape[0], 1).to(model.device)
    for t in range(pad_steps):
        r_out,_ = model.step(prex)
        model.prop(learn=False)
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


def test(model, x_test, y_test, answer_step, pad_steps, beta):
    test_size, n_steps, _ = x_test.shape
    with torch.no_grad():
        model.reset()
        prex = torch.zeros(test_size, 1).to(model.device)
        for t in range(pad_steps):
            r_out,_ = model.step(prex)
        pred_p = torch.zeros(test_size, 10).to(model.device)
        for t in range(n_steps):
            r_out,_ = model.step(x_test[:, t])
            if n_steps-t <= answer_step:
                p = torch.softmax(r_out, dim=1)
                pred_p += p*beta[t]
        prediction = torch.argmax(pred_p, dim=1)
        one_hot_label = F.one_hot(y_test, num_classes=10)
        loss = -(one_hot_label * torch.log(pred_p)).mean().item()
        acc_p = (prediction==y_test).sum().item()*100/test_size

    return acc_p, loss

def test_mul(model, x_test, y_test, answer_step, pad_steps, beta):
    #test differen ways to generate prediction
    test_size, n_steps, _ = x_test.shape
    with torch.no_grad():
        model.reset()
        prex = torch.zeros(test_size, 1).to(model.device)
        for t in range(pad_steps):
            r_out,_ = model.step(prex)
        pred_p = torch.zeros(test_size, 10).to(model.device)
        pred_r = torch.zeros(test_size, 10).to(model.device)
        pred_m = torch.zeros(test_size, 10).to(model.device)
        for t in range(n_steps):
            r_out,_ = model.step(x_test[:, t])
            if n_steps-t <= answer_step:
                p = torch.softmax(r_out, dim=1)
                pred_r += r_out*beta[t]
                pred_p += p*beta[t]
                pred_m[torch.arange(test_size), torch.argmax(p, dim=1)] += beta[t]
        prediction = torch.argmax(pred_p, dim=1)
        acc_p = (prediction==y_test).sum().item()*100/test_size
        acc_r = (torch.argmax(pred_r, dim=1)==y_test).sum().item()*100/test_size
        acc_m = (torch.argmax(pred_m, dim=1)==y_test).sum().item()*100/test_size

    return acc_p, acc_r, acc_m


def train_batch_BPTT(model, optimizer, x, y, answer_step, pad_steps, beta):
    n_steps = x.shape[1]
    model.reset()
    prex = torch.zeros(x.shape[0], 1).to(model.device)
    total_loss = 0.
    for t in range(pad_steps):
        r_out,_ = model.step(prex)
    for t in range(n_steps):
        r_out,_ = model.step(x[:, t])
        if n_steps-t <= answer_step:
            p = torch.softmax(r_out, dim=1)
            total_loss += -(y * torch.log(p)).mean()*beta[t]
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
            
    return total_loss.detach()


