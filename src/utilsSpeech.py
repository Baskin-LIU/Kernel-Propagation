import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Network import *
from inputFuc import *
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS


### DEFAULT Config ####
default_general_config = {
    'seed': 0, 
    'dt': 2., 
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
    'hop_length': 64,
    'n_mels': 48,
    'duration': 1000, #ms
    }

default_train_config = {
    'num_epochs': 150, 
    'learning_rate': 2e-2, 
    'batch_size': 256,
    'update_intervel': 200, #ms
    'num_workers': 4,
    "balance_sampler":True,
    }

default_model_config = {
    'n_in': 48, 
    'n_out': 35, 
    'num_LP_layers': 3, 
    'num_Ins_layers': 1, 
    'LP_size': [90, 120, 150], 
    'Ins_size': [150, ], 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': [default_general_config['dt'], 16, 6], 
    'Tau1': [4, 20], 
    'Tau2': [5, 24, 48],
    "answer_period": 800,
    }

from pathlib import Path
import torch


# class ShardedMelDataset(torch.utils.data.Dataset):
#     def __init__(self, split_dir):

#         split_dir = Path(split_dir)

#         with open(split_dir / "shard_index.json") as f:
#             shard_info = json.load(f)

#         self.shards = []
#         self.cum_sizes = []

#         total = 0
#         for s in shard_info:
#             mel = np.load(split_dir / s["mel_file"], mmap_mode="r")
#             lab = np.load(split_dir / s["label_file"], mmap_mode="r")
#             self.shards.append((mel, lab))
#             total += len(lab)
#             self.cum_sizes.append(total)

#     def __len__(self):
#         return self.cum_sizes[-1]

#     def __getitem__(self, idx):

#         shard_id = np.searchsorted(self.cum_sizes, idx, side="right")
#         prev = 0 if shard_id == 0 else self.cum_sizes[shard_id - 1]
#         local_idx = idx - prev

#         mel, lab = self.shards[shard_id]
#         return torch.from_numpy(mel[local_idx]), torch.tensor(lab[local_idx])


class ShardedMelDataset(torch.utils.data.Dataset):

    def __init__(self, split_dir):
        split_dir = Path(split_dir)

        with open(split_dir / "shard_index.json") as f:
            self.shard_info = json.load(f)

        self.split_dir = split_dir

        labels = []
        for s in self.shard_info:
            shard_labels = np.load(self.split_dir / s["label_file"])
            labels.append(shard_labels)
    
        self.labels = np.concatenate(labels, dtype=np.int16)

        # store only paths (pickle-safe)
        self.mel_paths = [split_dir / s["mel_file"] for s in self.shard_info]
        self.label_paths = [split_dir / s["label_file"] for s in self.shard_info]
        self.sizes = [s["num_samples"] for s in self.shard_info]

        # cumulative index map
        self.cum_sizes = np.cumsum(self.sizes)

        # memmaps will be opened lazily per worker
        self._mels = None
        self._labels = None

    def _lazy_init(self):
        if self._mels is None:
            self._mels = [
                np.load(p, mmap_mode="r") for p in self.mel_paths
            ]
            self._labels = [
                np.load(p, mmap_mode="r") for p in self.label_paths
            ]

    def __len__(self):
        return int(self.cum_sizes[-1])

    def __getitem__(self, idx):
        self._lazy_init()

        shard_id = np.searchsorted(self.cum_sizes, idx, side="right")
        prev = 0 if shard_id == 0 else self.cum_sizes[shard_id - 1]
        local_idx = idx - prev

        x = torch.tensor(self._mels[shard_id][local_idx])
        y = torch.tensor(self._labels[shard_id][local_idx])
        return x, y

class CollateMel:
    def __init__(self, target_time_steps: int, n_class: int):
        self.target_T = target_time_steps
        self.n_class = n_class

    def __call__(self, batch):
        """
        batch: list of (mel [n_mels, T], label)
        returns:
            mels  [B, n_mels, target_T]
            labels [B]
        """
        mels, labels = zip(*batch)

        resized = []
        for mel in mels:
            # mel: [n_mels, T]
            mel = mel.unsqueeze(0) # [1, n_mels, T]
            mel = F.interpolate(
                mel,
                size=self.target_T,
                mode="linear",
                align_corners=False
            )
            resized.append(mel.squeeze(0))

        mels = torch.stack(resized)
        labels = torch.tensor(labels, dtype=torch.long)
        OntHot = F.one_hot(labels, num_classes=self.n_class)

        return mels, OntHot


def train_batch_periodic(model, optimizer, x, y, answer_step, beta, update_period=10):
    n_steps = x.shape[-1]
    avg_p = 0.
    with torch.no_grad():
        model.reset()
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
    n_steps = x.shape[-1]
    avg_p = 0.
    with torch.no_grad():
        model.reset()
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


def test(model, data_loader, answer_step, beta):
    model.eval()
    num_sample=len(data_loader.dataset)
    loss, correct = 0., 0
    device = model.device
    with torch.no_grad():
        for x_test, y_test in data_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            n_steps = x_test.shape[-1]

            model.reset()
            pred_p = 0
            for t in range(n_steps):
                r_out,_ = model.step(x_test[:,:,t])
                if n_steps-t <= answer_step:
                    p = torch.softmax(r_out, dim=1)
                    pred_p = p*beta[t] + pred_p
            prediction = torch.argmax(pred_p, dim=1)
            loss += -(y_test * torch.log(pred_p)).mean(dim=1).sum().item()
            correct += (prediction==torch.argmax(y_test, dim=1)).sum().item()

    loss /= num_sample
    acc_p = correct*100/num_sample

    return acc_p, loss

def train_batch_BPTT(model, optimizer, x, y, answer_step, beta):
    n_steps = x.shape[-1]
    model.reset()
    total_loss = 0.
    for t in range(n_steps):
        r_out,_ = model.step(x[:, :, t])
        if n_steps-t <= answer_step:
            p = torch.softmax(r_out, dim=1)
            total_loss += -(y * torch.log(p)).mean(dim=1).sum()*beta[t]
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
            
    return total_loss.detach()


def make_weighted_sampler(dataset, label_counts):
    """
    dataset.labels : list or 1D tensor of class indices
    label_counts   : dict {class_id: count}
    """

    # inverse frequency per class
    class_weights = {
        LABELS.index(cls): 1.0 / count
        for cls, count in label_counts.items()
    }

    sample_weights = torch.tensor(
        [class_weights[int(label)] for label in dataset.labels],
        dtype=torch.double
    )

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


LABELS = ['backward', 'bed', 'bird',
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
    return torch.tensor(LABELS.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return LABELS[index]

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("../", download=True)

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


def collate_fn_wav(batch):
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

class MelDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.samples = list(self.root.rglob("*.pt"))

        self.labels = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path = self.samples[idx]
        mel = torch.load(pt_path)

        label = pt_path.parent.name
        label_idx = self.label_to_idx[label]

        return mel, label_idx