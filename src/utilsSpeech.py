import torch
import torch.nn.functional as F
import numpy as np
import os
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Network import *
from inputFuc import *
from torchaudio.datasets import SPEECHCOMMANDS


### DEFAULT Config ####
default_general_config = {
    'seed': 0, 
    'dt': 1., 
    'device': 'cpu', 
    'short_training_run': False,
    'visual_kernel': False,
    "verbose": True
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
    'batch_size': 128,
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
    'Tau0': [1, 12, 6], 
    'Tau1': [3, 12], 
    'Tau2': [5, 16],
    "answer_period":300,
    }

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("../data", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0., padding_side='right')
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def train_batch(model, optimizer, x, y, answer_step, beta, update_period=10):
    n_steps = x.shape[1]
    avg_p = 0.
    model.reset()
    #optimizer.zero_grad(set_to_none=False)
    prex = torch.zeros(x.shape[0], 1).to(model.device)
    for t in range(n_steps):
        r_out,_ = model.step(x[:,:,t])
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
    
    model.reset()
    prex = torch.zeros(x.shape[0], 1).to(model.device)
    for t in range(n_steps):
        r_out,_ = model.step(x[:,:,t])
        if n_steps-t <= answer_step:
            p = torch.softmax(r_out, dim=1)
            error = (y-p)*beta[t]
            model.prop(error=error)
            model.backwards()
            avg_p = p + avg_p
        else:
            model.prop(learn=False)
    optimizer.step()

    avg_p /= answer_step
    total_loss = -(y * torch.log(avg_p+1e-7)).mean(dim=1).sum().item()        
            
    return total_loss


def test(model, data_loader, transform, n_class, answer_step, beta):
    model.eval()
    num_sample=len(data_loader.dataset)
    loss, correct = 0., 0
    device = model.device
    with torch.no_grad():
        for x_test, y_test in data_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            x_test = transform(x_test)[:,0,:,:-1]
            n_steps = x_test.shape[-1]

            model.reset()
            pred_p = 0
            for t in range(n_steps):
                r_out,_ = model.step(x_test[:,:,t])
                if n_steps-t <= answer_step:
                    p = torch.softmax(r_out, dim=1)
                    pred_p = p*beta[t] + pred_p
            prediction = torch.argmax(pred_p, dim=1)
            one_hot_label = F.one_hot(y_test, num_classes=n_class)
            loss += -(one_hot_label * torch.log(pred_p)).mean(dim=1).sum().item()
            correct += (prediction==y_test).sum().item()

    loss /= num_sample
    acc_p = correct*100/num_sample

    return acc_p, loss



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

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]