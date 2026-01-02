import torch
import torch.nn.functional as F
import numpy as np
from FwdNeuron import *
from DeepEligNeuron import *
from Network import *
from inputFuc import *
from utils import *
from plotting import *


### DEFAULT Config ####
default_general_config = {
    'seed': 0, 
    'dt': 0.5, 
    'device': 'cpu', 
    'short_training_run': False,
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
    'final_seq_length': 72,
    'seed': 42,
    'url': 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl',
    'duration': 72,
    'pad': 1,      
    }

default_train_config = {
    'num_epochs': 100, 
    'learning_rate': 1e-4, 
    'batch_size': 40, 
    'num_workers': 4, 
    'num_prefetch_batch': 2,
    }

default_model_config = {
    'n_in': 1, 
    'n_out': 10, 
    'num_LP_layers': 4, 
    'num_Ins_layers': 1, 
    'LP_size': (90, 80, 80, 90), 
    'Ins_size': (60, 30), 
    'activation': 'tanh', 
    'Tau0': (1, 120), 
    'Tau1': np.array([1.2, 8. ]), 
    'Tau2': np.array([1.3, 8.1]),
    'Tau3': np.array([1.4, 8.2]),
    }

def buildMNISTNet(model_config, general_config):
    LP_size = list(model_config['LP_size'])
    Ins_size = list(model_config['Ins_size'])
    dt = general_config['dt']
    tau = []
    tau_min, tau_max = model_config['Tau0']
    tau.append(np.logspace(np.log10(tau_min), np.log10(tau_max), LP_size[0], dtype=np.float32))
    for i in range(model_config['num_LP_layers']-1):
        tau_uniq = model_config['Tau%d'%(i+1)]
        tau.append(np.repeat(tau_uniq[:, None],
                                          LP_size[i+1]//tau_uniq.shape[0]))
    layers = torch.nn.ModuleList()
    prev_n=model_config['n_in']
    
    # for i in range(model_config['num_LP_layers']-1):
    #     scale = 1. if i==0 else 0.2
    #     activation = model_config["activation"] if i==0 else "linear"
    #     layers.append(
    #         FwdDENeurons(
    #             n_in=prev_n,
    #             n_neurons=LP_size[i],
    #             tau=tau[i], 
    #             activation=activation, 
    #             dt=dt, 
    #             scale=scale
    #             )
    #     )
    #     prev_n=LP_size[i]

    layers.append(
        FwdDENeurons(
            n_in=prev_n,
            n_neurons=LP_size[0],
            tau=tau[0], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.
        )
    )
    prev_n=LP_size[0]

    for i in range(model_config['num_LP_layers']-2):
        layers.append(
            FwdDENeuronsReduced(
                n_in=prev_n,
                n_neurons=LP_size[i+1],
                tau=tau[i+1], 
                activation="linear", 
                dt=dt, 
            )
        )
        prev_n=LP_size[i+1]

    layers.append(
        LastFwdDENeurons(
            n_in=prev_n, 
            n_neurons=LP_size[i+2], 
            tau=tau[i+2], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.0
            )
    )
    prev_n=LP_size[i+2]
    for i in range(model_config['num_Ins_layers']):
        layers.append(
            FwdInsNeurons(
                n_in=prev_n,
                n_neurons=Ins_size[i],
                activation=model_config["activation"], 
                scale=1.0,
                dt=dt
            )
        )
        prev_n=Ins_size[i]
                
    layers.append(
        FwdInsNeurons(
            n_in=prev_n, 
            n_neurons=model_config['n_out'],
            activation='linear', 
            dt=dt, 
            scale=1.0
            )
    )
    
    return DEFwdNetwork(layers=layers)


def train_batch(model, optimizer, x, y, answer_period):
    n_steps = x.shape[1]
    total_error = 0.
    one_hot_label = F.one_hot(y, num_classes=10)
    model.reset()
    prex = torch.zeros(x.shape[0], 1)
    for t in range(10):
        r_out,_ = model.step(prex)
        model.prop(learn=False)
    for t in range(n_steps):
        r_out,_ = model.step(x[:, t])
        if n_steps-t <= answer_period:
            p = torch.softmax(r_out, dim=1)
            error = (one_hot_label - p)/answer_period
            model.prop(error=error)
            model.backwards()
            optimizer.step()
            optimizer.zero_grad()
            total_error += -(one_hot_label * torch.log(p)).mean().item()
        else:
            model.prop(learn=False)
            
            
    return total_error/answer_period


def test(model, x_test, y_test, answer_period=2):
    test_size, n_steps, _ = x_test.shape
    with torch.no_grad():
        model.reset()
        pred = torch.zeros(test_size, 10)
        for t in range(n_steps):
            r_out,_ = model.step(x_test[:, t])
            if n_steps-t <= answer_period:
                p = torch.softmax(r_out, dim=1)
                pred += p
        pred /= answer_period
        prediction=torch.argmax(pred, dim=1)
        one_hot_label = F.one_hot(y_test, num_classes=10)
        loss = -(one_hot_label * torch.log(pred)).mean().item()
        acc = ((prediction==y_test)*1.).mean().item()
    return acc, loss