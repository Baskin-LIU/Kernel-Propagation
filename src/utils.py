import torch
import torch.nn.functional as F
import numpy as np
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Network import *
from inputFuc import *

def buildKPNet(model_config, general_config):
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

    if model_config["version"]=="V2":
        Neurons = FwdDENeuronsV2
        print("rho bar")
    else:
        Neurons = FwdDENeurons
    
    layers.append(
        Neurons(
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
                Neurons(
                    n_in=prev_n,
                    n_neurons=LP_size[i+1],
                    tau=tau[i+1], 
                    activation=model_config["activation"], 
                    dt=dt, 
                    scale=model_config["rho_scale"],# 0.6,
                    device=device,
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

    if 'skip_connection' in model_config:
        if model_config['skip_connection']=='All':
            model = DEAllSkipNetwork(layers=layers, device=device, dt=dt)
            print("All skip connections activated")
        elif model_config['skip_connection']=="One":
            model = DESkipNetwork(layers=layers, device=device, dt=dt)
            print("A skip connection activated")
        else:
            model = DEFwdNetwork(layers=layers, device=device, dt=dt)
    else:
        model = DEFwdNetwork(layers=layers, device=device, dt=dt)

    if 'upsample' in model_config and model_config["upsample"]:
        print("Upsample Input")
        with torch.no_grad():
            model.layers[0].W_in /= model.layers[0].W_in
            model.layers[0].bias *= 0
            weights = [0.05, -0.1, 0.5, -1.0, -1.5, 3.]*model.layers[0].n_tau
            model.layers[0].W_in *= torch.tensor(weights)[:, None]
            model.layers[1].rho.scale = 1.
            model.learn_depth = 1
    return model


def buildNetCompare(model_config, general_config, neurontype='GLE'):
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
    if neurontype=='GLE' or neurontype=='LE':
        layer_fn = FwdGLENeurons  
    elif neurontype=='RF/E':
        layer_fn = FwdRFNeurons
    elif neurontype=='BPTT':
        layer_fn = FwdNeurons
    else:
        raise NotImplementedError

    prev_n=model_config['n_in']
    for i in range(model_config['num_LP_layers']):
        if i==0 or i+1==model_config['num_LP_layers']:
            scale = 1.
            activation = model_config["activation"]
        elif model_config["reducedNonlinear"]:
            scale=1.
            activation = "linear"
        else:
            scale=0.6
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
        
    # if 'skip_connection' in model_config and model_config['skip_connection']:
    #     model = SkipNetwork(layers=layers, dt=dt, device=device)
    #     print("skip connection activated")
    # else:
    #     model = FwdNetwork(layers=layers, dt=dt, device=device)

    if 'skip_connection' in model_config:
        if model_config['skip_connection']=='All':
            model = AllSkipNetwork(layers=layers, device=device, dt=dt)
            print("All skip connections activated")
        elif model_config['skip_connection']=="One":
            model = SkipNetwork(layers=layers, device=device, dt=dt)
            print("A skip connection activated")
        else:
            model = FwdNetwork(layers=layers, device=device, dt=dt)
    else:
        model = DEFwdNetwork(layers=layers, device=device, dt=dt)

    if 'upsample' in model_config and model_config["upsample"]:
        print("Upsample Input")
        with torch.no_grad():
            model.layers[0].W_in /= model.layers[0].W_in
            model.layers[0].bias *= 0
            _, _, repeat = model_config['Tau0']
            weights = [0.05, -0.1, 0.5, -1.0, -1.5, 3.]*(model.layers[0].n_neurons//repeat)
            model.layers[0].W_in *= torch.tensor(weights)[:, None]
            model.layers[1].rho.scale = 1.
            model.learn_depth = 1

    if neurontype=='LE':
        for l in model.layers[:model_config['num_LP_layers']]:
            l.tau_dt *= 0
    if neurontype=="RF/E":
        model.learn_depth = model_config['num_LP_layers']-1
    if neurontype=="BPTT":
        for l in model.layers[:model.learn_depth]:
            l.W_in.requires_grad = False
            l.bias.requires_grad = False
    
    return model
    

class Recorder():

    def __init__(self, names, dt=1.):
        self.record = {name:[] for name in names}
        self.dt = dt

    def rec(self, data):
        for i, rcd in enumerate(self.record.values()):
            rcd.append(data[i].detach().clone())

    def rec_single(self, name, data):
        self.record[name].append(data.detach().clone())
            
    def finish(self,):
        for name, rcd in self.record.items():
            try:
                self.record[name] = torch.vstack(rcd).numpy().T
                self.steps=self.record[name].shape[1]
            except:
                continue

    def __getitem__(self, name):
        # This allows: recorder[name]
        return self.record[name]

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
        