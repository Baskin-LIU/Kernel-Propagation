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
    'prepad': 20,      
    }

default_train_config = {
    'num_epochs': 100, 
    'learning_rate': 1e-2, 
    'batch_size': 100, 
    "answer_period":300,
    }

default_model_config = {
    'n_in': 1, 
    'n_out': 10, 
    'num_LP_layers': 3, 
    'num_Ins_layers': 1, 
    'LP_size': (64, 96, 96), 
    'Ins_size': (150, ), 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': (1, 20, 2), 
    'Tau1': (2, 6., 6., 6., 16., 16.), 
    'Tau2': (1, 8., 8., 8., 32., 32.),
    }

def buildMNISTNet(model_config, general_config):
    device = general_config["device"]
    LP_size = list(model_config['LP_size'])
    Ins_size = list(model_config['Ins_size'])
    dt = general_config['dt']
    tau = []
    tau_min, tau_max, repeat = model_config['Tau0']
    tau0 = np.logspace(np.log10(tau_min), np.log10(tau_max), LP_size[0]//repeat, dtype=np.float32)
    tau.append(np.repeat(tau0[:, None], repeat))
    for i in range(model_config['num_LP_layers']-1):
        tau_uniq = np.array(model_config['Tau%d'%(i+1)])
        tau.append(np.repeat(tau_uniq[:, None],
                                          LP_size[i+1]//tau_uniq.shape[0]))
    layers = torch.nn.ModuleList()
    prev_n=model_config['n_in']
    
    layers.append(
        FwdDENeurons(
            n_in=prev_n,
            n_neurons=LP_size[0],
            tau=tau[0], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.0,
            device=device,
        )
    )
    prev_n=LP_size[0]

    if model_config["reducedNonlinear"]:
        for i in range(model_config['num_LP_layers']-2):
            layers.append(
                FwdDENeuronsReduced(
                    n_in=prev_n,
                    n_neurons=LP_size[i+1],
                    tau=tau[i+1], 
                    activation="linear", 
                    dt=dt, 
                    device=device,
                )
            )
            prev_n=LP_size[i+1]
    else:
        for i in range(model_config['num_LP_layers']-2):
            layers.append(
                FwdDENeurons(
                    n_in=prev_n,
                    n_neurons=LP_size[i+1],
                    tau=tau[i+1], 
                    activation=model_config["activation"], 
                    dt=dt, 
                    scale=0.9,
                    device=device,
                )
            )
            prev_n=LP_size[i+1]

    # layers.append(
    #     FwdDENeuronsReduced(
    #         n_in=prev_n,
    #         n_neurons=prev_n,
    #         tau=tau[i+1]+0.1, 
    #         activation="linear", 
    #         dt=dt, 
    #         device=device,
    #     )
    # )

    layers.append(
        LastFwdDENeurons(
            n_in=prev_n, 
            n_neurons=LP_size[i+2], 
            tau=tau[i+2], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.0,
            device=device,
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
                dt=dt,
                device=device,
            )
        )
        prev_n=Ins_size[i]
                
    layers.append(
        FwdInsNeurons(
            n_in=prev_n, 
            n_neurons=model_config['n_out'],
            activation='linear', 
            dt=dt, 
            scale=1.0,
            device=device,
            )
    )
    
    return DEFwdNetwork(layers=layers, device=device)


def train_batch(model, optimizer, x, y, answer_step, pad_steps, beta):
    n_steps = x.shape[1]
    avg_p = 0.
    one_hot_label = F.one_hot(y, num_classes=10)
    model.reset()
    prex = torch.zeros(x.shape[0], 1).to(model.device)
    for t in range(pad_steps):
        r_out,_ = model.step(prex)
        model.prop(learn=False)
    for t in range(n_steps):
        r_out,_ = model.step(x[:, t])
        if n_steps-t <= answer_step:
            p = torch.softmax(r_out, dim=1)
            error = (one_hot_label - p)*beta[t]#/answer_period
            optimizer.zero_grad(set_to_none=False)
            model.prop(error=error)
            model.backwards()
            optimizer.step()
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
    optimizer.zero_grad(set_to_none=False)
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


def test(model, x_test, y_test, answer_step, start_step, beta):
    test_size, n_steps, _ = x_test.shape
    with torch.no_grad():
        model.reset()
        prex = torch.zeros(test_size, 1).to(model.device)
        for t in range(start_step):
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
        acc_p = ((prediction==y_test)*1.).mean().item()*100

    return acc_p, loss


def extract_kernel(model, n_steps, layer_idx):
    model.reset()
    impulse = torch.zeros(1, n_steps, 1).to(model.device)
    impulse[0] += 1
    kernel_record = []
    for t in range(n_steps):
        r = impulse[:, t]
        for l in model.layers[:layer_idx]:
            r,u = l.step(r)
        kernel_record.append(u.detach().clone())

    return torch.vstack(kernel_record).cpu().numpy().T



def buildMNISTNetCompare(model_config, general_config, neurontype='GLE'):
    device = general_config["device"]
    LP_size = list(model_config['LP_size'])
    Ins_size = list(model_config['Ins_size'])
    dt = general_config['dt']
    tau = []
    tau_min, tau_max = model_config['Tau0']
    tau0 = np.logspace(np.log10(tau_min), np.log10(tau_max), LP_size[0]//3, dtype=np.float32)
    tau.append(np.repeat(tau0[:, None], 3))
    for i in range(model_config['num_LP_layers']-1):
        tau_uniq = model_config['Tau%d'%(i+1)]
        tau.append(np.repeat(tau_uniq[:, None],
                                          LP_size[i+1]//tau_uniq.shape[0]))
    layers = torch.nn.ModuleList()
    layer_fn = FwdGLENeurons if neurontype=='GLE' else FwdRFNeurons

    prev_n=model_config['n_in']
    for i in range(model_config['num_LP_layers']):
        if i==0 or i+1==model_config['num_LP_layers']:
            scale = 1.
            activation = model_config["activation"]
        elif model_config["reducedNonlinear"]:
            scale=1.
            activation = "linear"
        else:
            scale=0.4
            activation = model_config["activation"]


        layers.append(
            layer_fn(
                n_in=prev_n,
                n_neurons=LP_size[i],
                tau=tau[i], 
                activation=activation, 
                dt=dt, 
                scale=scale,
                device=device,
            )
        )
        prev_n=LP_size[i]

    prev_n=LP_size[i+2]
    for i in range(model_config['num_Ins_layers']):
        layers.append(
            FwdInsNeurons(
                n_in=prev_n,
                n_neurons=Ins_size[i],
                activation=model_config["activation"], 
                scale=1.0,
                dt=dt,
                device=device,
            )
        )
        prev_n=Ins_size[i]
                
    layers.append(
        FwdInsNeurons(
            n_in=prev_n, 
            n_neurons=model_config['n_out'],
            activation='linear', 
            dt=dt, 
            scale=1.0,
            device=device,
            )
    )
    
    return FwdNetwork(layers=layers, dt=dt, device=device)