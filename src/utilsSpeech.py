import torch
import torch.nn.functional as F
import numpy as np
import numpy.random as random
import os
import json
from pathlib import Path
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Network import *
from inputFuc import *
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS

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

COMMAND20 = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'nine', 'eight',
             'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'Unknown']

INDEX20 = [20, 20, 20, 20, 20, 13, 9, 5, 20, 20, 4, 19, 20, 20, 20, 14, 20, 8, 11, 17, 16, 1, 15, 7, 20, 6, 18, 3, 20, 2, 12, 20, 20, 10, 0]


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
    'hop_length': 80,
    'n_mels': 64,
    'duration': 1000, #ms
    }

default_train_config = {
    'num_epochs': 150, 
    'learning_rate': 2e-3, 
    'batch_size': 256,
    'update_interval': 200, #ms
    'num_workers': 4,
    "balance_sampler":True,
    }

default_model_config = {
    'n_in': 48, 
    'n_out': 35, 
    'num_LP_layers': 3, 
    'num_Ins_layers': 1,
    'rho_scale': 0.6,
    'LP_size': [90, 120, 150], 
    'Ins_size': [150, ], 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': [default_general_config['dt'], 50, 6], 
    'Tau1': [3, 12, 24], 
    'Tau2': [4, 16, 48],
    "answer_period": 800,
    }


class ShardedMelDataset(torch.utils.data.Dataset):

    def __init__(self, split_dir):
        split_dir = Path(split_dir)

        with open(split_dir / "shard_index.json") as f:
            self.shard_info = json.load(f)

        self.split_dir = split_dir

        labels = []
        for s in self.shard_info:
            shard_labels = np.load(self.split_dir / s["label_file"])

            Shard_labels = []
            for label in shard_labels:
                Shard_labels.append(INDEX20[int(label)])
            shard_labels = np.array(Shard_labels)
            
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
        y = self._labels[shard_id][local_idx]
        y = torch.tensor(INDEX20[y])
        return x, y

class CollateMel:
    def __init__(self, target_time_steps: int, n_class: int, training=False, 
                 max_shift=3, mask_width=0, mask_width_fre=4, max_warp=10,):
        self.target_T = target_time_steps
        self.n_class = n_class

        self.training = training
        self.max_shift = max_shift
        self.mask_width = mask_width
        self.mask_width_fre = mask_width_fre
        self.max_warp = max_warp

    def _augment(self, mel):
        n_mel, T = mel.shape
        mean_val = mel.mean(dim=1, keepdim=True)  # [n_mels, 1]
        #Temporal mask
        #start = random.randint(0, T - self.mask_width)
        #mel[:, start:start + self.mask_width] = mean_val

        #Frequency mask
        start = random.randint(0, n_mel - self.mask_width_fre)
        mel[start:start + self.mask_width_fre, :] = mean_val[start:start + self.mask_width_fre]
        
        #Temporal shift
        #shift = random.randint(-self.max_shift, self.max_shift)
        # if shift == 0:
        #     return mel
        # if shift > 0:
        #     # shift right
        #     pad = mean_val.expand(-1, shift)
        #     mel = torch.cat([pad, mel[:, :-shift]], dim=1)
        # else:
        #     # shift left
        #     shift = abs(shift)
        #     pad = mean_val.expand(-1, shift)
        #     mel = torch.cat([mel[:, shift:], pad], dim=1)

        shift_left = random.randint(0, self.max_shift)
        shift_right = random.randint(1, self.max_shift)
        # pick pivot away from borders
        center = random.randint(low=80, high=120)
    
        # random displacement
        warp = random.randint(-self.max_warp, self.max_warp + 1)
        new_center = int(center*2.5) + warp
    
        # split
        left = mel[:, shift_left:center]
        right = mel[:, center:-shift_right]

        # resample each part
        left = F.interpolate(
            left.unsqueeze(0),
            size=new_center,
            mode="linear",
            align_corners=False
        )
    
        right = F.interpolate(
            right.unsqueeze(0),
            size=self.target_T - new_center,
            mode="linear",
            align_corners=False
        )
        mel = torch.cat([left, right], dim=-1)

        return mel

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
            if self.training:
                mel = self._augment(mel)
            else:
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


def test_analy(model, data_loader, answer_step, beta, num_classes=35, eps=1e-12):
    model.eval()
    num_sample = len(data_loader.dataset)
    device = model.device

    loss, correct = 0., 0

    # confusion stats
    TP = torch.zeros(num_classes, device=device)
    FP = torch.zeros(num_classes, device=device)
    FN = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for x_test, y_test in data_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_true = torch.argmax(y_test, dim=1)
            n_steps = x_test.shape[-1]

            model.reset()
            pred_p = 0.

            for t in range(n_steps):
                r_out, _ = model.step(x_test[:, :, t])
                if n_steps - t <= answer_step:
                    p = torch.softmax(r_out, dim=1)
                    pred_p = pred_p + p * beta[t]

            prediction = torch.argmax(pred_p, dim=1)

            # ---- loss and accuracy ----
            loss += -(y_test * torch.log(pred_p + eps)).mean(dim=1).sum().item()
            correct += (prediction == y_true).sum().item()

            # ---- per-class stats ----
            for c in range(num_classes):
                pred_is_c = (prediction == c)
                true_is_c = (y_true == c)

                TP[c] += (pred_is_c & true_is_c).sum()
                FP[c] += (pred_is_c & ~true_is_c).sum()
                FN[c] += (~pred_is_c & true_is_c).sum()

    loss /= num_sample
    acc_p = correct * 100 / num_sample

    # ---- metrics ----
    recall = TP / (TP + FN + eps)
    precision = TP / (TP + FP + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    per_class = {
        "TP": TP.cpu(),
        "FP": FP.cpu(),
        "FN": FN.cpu(),
        "recall": recall.cpu(),
        "precision": precision.cpu(),
        "f1": f1.cpu(),
    }

    return acc_p, loss, per_class

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
    # class_weights = {
    #     LABELS.index(cls): 1.0 / count
    #     for cls, count in label_counts.items()
    # }

    # sample_weights = torch.tensor(
    #     [class_weights[int(label)] for label in dataset.labels],
    #     dtype=torch.double
    # )

    class_number = {
        LABELS.index(cls): count
        for cls, count in label_counts.items()
    }

    class_number20 = torch.zeros(21)
    for word in class_number.keys():
        class_number20[INDEX20[word]] += class_number[word]
    class_weights = 1.0/class_number20
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