import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import sys
sys.path.insert(0, "../src/")
from FwdNeuron import *
from DeepEligNeuron import *
from Network import *
from inputFuc import *
from utils import *
from plotting import *

from mnist1d.data import make_dataset, get_dataset_args, get_templates
from mnist1d.utils import from_pickle, to_pickle, ObjectView, set_seed, plot_signals


def train(model, samples):
    batch, n_steps, _ = samples.shape

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ### General config
    parser.add_argument("--short_run", dest="short_run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.5) #ms
    #parser.add_argument("--machine", type=str, default="MLcloud")
    
    ### Training config
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)

    ### Data config
    parser.add_argument("--duration", type=float, default=72) #ms
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_prefetch_batch", type=int, default=2)
    
    ### Model config
    parser.add_argument("--num_LP_layers", type=int, default=3)
    parser.add_argument("--num_Ins_layers", type=int, default=1)
    parser.add_argument("--LP_size", type=tuple, default=10)
    parser.add_argument("--Ins_size", type=tuple, default=1)
    parser.add_argument("--activation", type=str, default='tanh')
    parser.add_argument("--Tau0", type=tuple, default=30.)
    parser.add_argument("--Tau", type=tuple, default=30.)
    
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    
    parser.set_defaults(short_run=False, )
    
    args = parser.parse_args()

    ########## General Config ##########
    print("General configuration started...")

    # General Config
    general_config = dict()
    general_config["seed"] = args.seed
    general_config["dt"] = args.dt
    general_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    general_config["short_training_run"] = args.short_run
    torch_device = torch.device(general_config["device"])
    print("Torch Device: ", torch_device)

    # Seeding & Determinism
    os.environ["PYTHONHASHSEED"] = str(general_config["seed"])
    random.seed(general_config["seed"])
    np.random.seed(general_config["seed"])
    torch.manual_seed(general_config["seed"])
    torch.cuda.manual_seed(general_config["seed"])
    torch.backends.cudnn.deterministic = True

    ########## Data, Model and Training Config ##########
    print("Data, model and training configuration started...")

    # Model Config
    model_config = dict()
    model_config["n_in"] = 1
    model_config["n_out"] = 10
    model_config["num_LP_layers"] = args.num_LP_layers
    model_config["num_Ins_layers"] = args.num_Ins_layers
    model_config["LP_size"] = args.LP_size
    model_config["Ins_size"] = args.Ins_size
    model_config["activation"] = args.activation
    model_config["Tau0"] = args.Tau0
    Tau = np.array(args.Tau)
    for i in range(model_config["num_LP_layers"]-1):
        model_config["Tau%d"%(i+1)] = Tau + i*0.1

    # Training Config
    train_config = dict()
    train_config["num_epochs"] = 1 if general_config["short_training_run"] else args.num_epochs
    train_config["learning_rate"] = args.lr
    
    
    # Init Dataset
    data_config = {
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
        'final_seq_length': 144,
        'seed': 42,
        'url': 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
    }

    data_config = dict()
    data_config["duration"] = args.duration
    data_config["pad"] = args.pad
    data_config["batch_size"] = 20 if general_config["short_training_run"] else args.batch
    data_config["num_workers"] = args.num_workers # will make run nondeterministic
    data_config["num_prefetch_batch"] = args.num_prefetch_batch
    data = make_dataset(ObjectView(data_config))
    x, y = torch.tensor(data['x'],dtype=torch.float32).unsqueeze(-1), torch.tensor(data['y'])

    model = DENetwork(**model_config)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10
    )

    with torch.no_grad():
        for epoch in range(num_epochs):
            permu = np.random.permutation(4000).reshape(-1, batch)
            for i, idx in enumerate(permu):
                student.reset(batch)
                total_error = 0.
                one_hot_label = torch.zeros(batch, 10)   
                one_hot_label[np.arange(batch), y[idx]] = 1.
                sample = x[idx]
                for t in range(n_steps):
                    r_out, u_out = model.step(sample[:, t])
                    if t>=136:
                        p = torch.softmax(r_out, dim=1)
                        error = one_hot_label - p
                        student.prop(error)
                        student.backwards()
                        optimizer.step()
                        optimizer.zero_grad()
                        total_error += -(one_hot_label * torch.log(p)).mean().item()
                    else:
                        student.prop(0)
                
                Cum_errors = 0.98*Cum_errors + 0.02*total_error/8


                #TEST
                    x_test, y_test = torch.tensor(data['x_test'],dtype=torch.float32).unsqueeze(-1), torch.tensor(data['y_test'])
                    prediction = []
                    acc = 0.
                    with torch.no_grad():
                        permu = np.random.permutation(1000).reshape(-1, batch)
                        for i, idx in enumerate(permu):
                            sample = x[idx]
                            pred = torch.zeros(batch, 10)
                            for t in range(n_steps):
                                r_out,_= student.step(sample[:, t])
                                if t>130:
                                    pred += r_out
                            student.reset()
                            prediction=torch.argmax(pred, dim=1)
                            acc += ((prediction.numpy()==y[idx])*1.).sum()
                    acc /= 1000
                    print('Accuracy: ', acc)
                if (i*batch)%1000==0:
                    scheduler.step(Cum_errors)
                    error_record.append(Cum_errors)   
                    print(" error: %5e, iteration: %d samples"%(Cum_errors, i*batch))