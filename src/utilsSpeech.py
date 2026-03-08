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

COMMAND20 = ['zero', 'one', 'two', 'three', 'four', 
             'five', 'six', 'seven', 'nine', 'eight',
             'yes', 'no', 'up', 'down', 'left', 'right', 
             'on', 'off', 'stop', 'go']

WORDS1600 = ["bird","cat","dog","down","eight","five","four","go","happy","house","left","marvin","nine","no",
             "off","on","one","right","seven","sheila","six","stop","three","two","up","wow","yes","zero"]

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
    'n_class':21, 
    'sample_rate':16000,
    'preprocessing':'Mel',
    'n_fft': 400,
    'win_length': 400,
    'hop_length': 100,
    'n_mels': 80,
    'duration': 1000, #ms
    }

default_train_config = {
    'num_epochs': 150, 
    'learning_rate': 2e-3, 
    'batch_size': 256,
    'update_interval': 200, #ms
    'num_workers': 4,
    "weighted_sampler":False,
    "error_steps":0,
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

    def __init__(self, split_dir, task='Cmd20'):
        if task == 'Cmd20':
            self.indexInFull = {LABELS.index(w): i for i, w in enumerate(COMMAND20)}
        elif task == 'WD1600':
            self.indexInFull = {LABELS.index(w): i for i, w in enumerate(WORDS1600)}
        split_dir = Path(split_dir)

        with open(split_dir / "shard_index.json") as f:
            self.shard_info = json.load(f)

        self.split_dir = split_dir
        self.task = task

        self.mel_paths = []
        self.label_paths = []
        self.keep_indices = []   # local indices per shard
        self.sizes = []
        all_labels = []

        for s in self.shard_info:
            label_path = self.split_dir / s["label_file"]
            labels = np.load(label_path)

            if task != 'Full':
                mask = np.isin(labels, list(self.indexInFull.keys()))
                keep_idx = np.where(mask)[0]
                if len(keep_idx) == 0:
                    continue
                self.keep_indices.append(keep_idx)
                self.sizes.append(len(keep_idx))
                all_labels.append(np.array([self.indexInFull[lb] for lb in labels[keep_idx]]))
            else:
                self.keep_indices.append(None)
                self.sizes.append(len(labels))
                all_labels.append(labels)

            self.mel_paths.append(self.split_dir / s["mel_file"])
            self.label_paths.append(label_path)

        self.labels = np.concatenate(all_labels, dtype=np.int16)
        self.cum_sizes = np.cumsum(self.sizes)

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

        if self.task != 'Full':
            true_idx = self.keep_indices[shard_id][local_idx]
        else:
            true_idx = local_idx

        x = torch.tensor(self._mels[shard_id][true_idx])
        #y = self._labels[shard_id][true_idx]
        # if self.task != 'Full':
        #     y = torch.tensor(self.indexInFull[int(y)])
        # else:
        #     y = torch.tensor(y)
        y = torch.tensor(self.labels[idx])

        return x, y
        

class CollateMel:
    def __init__(self, target_time_steps: int, n_class: int, ori_len=160, training=False, 
                 max_shift=25, mask_width=0, mask_width_fre=4, max_warp=10,):
        self.target_T = target_time_steps
        self.n_class = n_class

        self.training = training
        self.max_shift = max_shift
        self.mask_width = mask_width
        self.mask_width_fre = mask_width_fre
        self.max_warp = max_warp

        self.interpo_ratio = self.target_T/ori_len
        self.warp_low = int(0.4 * ori_len)
        self.warp_high = int(0.6 * ori_len)

    def _augment(self, mel):
        n_mel, T = mel.shape
        mean_val = mel.mean(dim=1, keepdim=True)  # [n_mels, 1]
        #Temporal mask
        #start = random.randint(0, T - self.mask_width)
        #mel[:, start:start + self.mask_width] = mean_val

        mel *= torch.rand(1)*0.4+0.8
        std = mel.std() * 0.1

        #Frequency mask
        start = random.randint(0, n_mel - self.mask_width_fre)
        mel[start:start + self.mask_width_fre, :] = mean_val[start:start + self.mask_width_fre]

        #Temporal shift
        shift_left = random.randint(-self.max_shift*2, self.max_shift)
        shift_right = random.randint(-self.max_shift*2, self.max_shift)
        # pick pivot away from borders
        center = random.randint(low=self.warp_low, high=self.warp_high)
    
        # random displacement
        warp = random.randint(-self.max_warp, self.max_warp + 1)
        new_center = int(center*self.interpo_ratio) + warp
    
        # split
        if shift_left >=0:
            left = mel[:, shift_left:center]
        else:
            left = F.pad(mel[:, :center], (-shift_left, 0), mode="replicate")
        if shift_right >=1:
            right = mel[:, center:-shift_right]
        else:
            right = F.pad(mel[:, center:], (0, 1-shift_right), mode="replicate")
        left += torch.randn_like(left) * std
        right += torch.randn_like(right) * std

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
                if beta[t] != 0:
                    error = (y-p)*beta[t] 
                    model.prop(error=error)
                    model.backwards()
                else:
                    model.prop(learn=False)
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


def test_analy_confusion(model, data_loader, answer_step, beta, num_classes=None, eps=1e-12):
    model.eval()
    device = model.device
    num_sample = len(data_loader.dataset)

    loss, correct = 0., 0
    correct_r = 0.

    # infer class number
    if num_classes is None:
        num_classes = data_loader.dataset[0][1].shape[-1]

    # initialize confusion matrix
    confusion = torch.zeros(num_classes, num_classes, device=device, dtype=int)
    Mistakes = []
    index = 0

    with torch.no_grad():
        for x_test, y_test in data_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_true = torch.argmax(y_test, dim=1)
            n_steps = x_test.shape[-1]
            batch = x_test.shape[0]

            model.reset()
            pred_p = 0.
            r_out_sum = torch.zeros(batch, num_classes)

            for t in range(n_steps):
                r_out, _ = model.step(x_test[:, :, t])
                if n_steps - t <= answer_step:
                    p = torch.softmax(r_out, dim=1)
                    r_out_sum += r_out*beta[t]
                    pred_p = pred_p + p * beta[t]

            prediction = torch.argmax(pred_p, dim=1)
            prediction_r = torch.argmax(r_out_sum, dim=1)

            # ----- loss & accuracy -----
            loss += -(y_test * torch.log(pred_p + eps)).mean(dim=1).sum().item()
            correct += (prediction == y_true).sum().item()
            wrong = torch.nonzero(prediction != y_true, as_tuple=True)[0].numpy()
            Mistakes.append(wrong+index)
            index += batch

            correct_r += (prediction_r == y_true).sum().item()

            # ----- confusion matrix accumulation -----
            # flatten pairs (true, pred) into linear index
            indices = y_true * num_classes + prediction
            cm_batch = torch.bincount(
                indices,
                minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

            confusion += cm_batch

    loss /= num_sample
    acc_p = correct * 100 / num_sample

    acc_r = correct_r * 100 / num_sample

    # ---- derive metrics from confusion matrix ----
    TP = confusion.diag()
    FN = confusion.sum(dim=1) - TP
    FP = confusion.sum(dim=0) - TP
    TN = confusion.sum() - (TP + FP + FN)

    recall = TP / (TP + FN + eps)
    precision = TP / (TP + FP + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    results = {
        "acc_r": acc_r,
        "confusion_matrix": confusion.cpu().numpy(),
        "TP": TP.cpu().numpy(),
        "FP": FP.cpu().numpy(),
        "FN": FN.cpu().numpy(),
        "TN": TN.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "Mistakes": np.hstack(Mistakes)
    }

    return acc_p, loss, results


def train_batch_BPTT(model, optimizer, x, y, answer_step, beta):
    n_steps = x.shape[-1]
    model.reset()
    #class_weight = torch.ones(21).to(model.device)
    #class_weight[-1] = 0.6
    total_loss = 0.
    for t in range(n_steps):
        r_out,_ = model.step(x[:, :, t])
        if n_steps-t <= answer_step:
            p = torch.softmax(r_out, dim=1)
            total_loss += -(y*torch.log(p)).mean(dim=1).sum()*beta[t]
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
            
    return total_loss.detach()


def make_weighted_sampler(dataset, num_classes):
    """
    dataset.labels : list or 1D tensor of class indices
    label_counts   : dict {class_id: count}
    """

    class_counts = np.bincount(dataset.labels, minlength=num_classes)

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[dataset.labels]

    # #inverse frequency per class
    # class_weights = {
    #     LABELS.index(cls): 1.0 / count
    #     for cls, count in label_counts.items()
    # }

    # sample_weights = torch.tensor(
    #     [class_weights[int(label)] for label in dataset.labels],
    #     dtype=torch.double
    # )

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


# class ShardedMelDatasetUnknown(torch.utils.data.Dataset):

#     def __init__(self, split_dir, task='Cmd20'):
#         split_dir = Path(split_dir)

#         with open(split_dir / "shard_index.json") as f:
#             self.shard_info = json.load(f)

#         self.split_dir = split_dir
#         self.task = task

#         labels = []
#         for s in self.shard_info:
#             shard_labels = np.load(self.split_dir / s["label_file"])
#             labels.append(shard_labels)
    
#         self.labels = np.concatenate(labels, dtype=np.int16)

#         # store only paths (pickle-safe)
#         self.mel_paths = [split_dir / s["mel_file"] for s in self.shard_info]
#         self.label_paths = [split_dir / s["label_file"] for s in self.shard_info]
#         self.sizes = [s["num_samples"] for s in self.shard_info]

#         # cumulative index map
#         self.cum_sizes = np.cumsum(self.sizes)

#         # memmaps will be opened lazily per worker
#         self._mels = None
#         self._labels = None

#     def _lazy_init(self):
#         if self._mels is None:
#             self._mels = [
#                 np.load(p, mmap_mode="r") for p in self.mel_paths
#             ]
#             self._labels = [
#                 np.load(p, mmap_mode="r") for p in self.label_paths
#             ]

#     def __len__(self):
#         return int(self.cum_sizes[-1])

#     def __getitem__(self, idx):
#         self._lazy_init()

#         shard_id = np.searchsorted(self.cum_sizes, idx, side="right")
#         prev = 0 if shard_id == 0 else self.cum_sizes[shard_id - 1]
#         local_idx = idx - prev

#         x = torch.tensor(self._mels[shard_id][local_idx])
#         y = self._labels[shard_id][local_idx]
#         if self.task=='Full':
#             y = torch.tensor(y)
#         elif self.task=='Cmd20':
#             y = torch.tensor(INDEX20[y])
#         return x, y