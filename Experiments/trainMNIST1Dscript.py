import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
import wandb
import gc
import json
import random
import tempfile
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "../src/")
from Network import *
from utils import *
from plotting import *
from utilsMNIST import *

from mnist1d.data import make_dataset, get_dataset_args, get_templates
from mnist1d.utils import from_pickle, to_pickle, ObjectView, set_seed, plot_signals
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--group_name", type=str, default="test")
    ### General config
    parser.add_argument("--short_run", dest="short_run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.) #ms
    #parser.add_argument("--machine", type=str, default="MLcloud")
    
    ### Training config
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=160)
    parser.add_argument("--answer_t", type=int, default=300)

    ### Data config
    parser.add_argument("--duration", type=float, default=72) #ms
    parser.add_argument("--start_t", type=int, default=4)
    parser.add_argument("--batch", type=int, default=100)
    
    ### Model config
    parser.add_argument("--num_LP_layers", type=int, default=3)
    parser.add_argument("--num_Ins_layers", type=int, default=1)
    parser.add_argument("--LP_size", type=tuple, default=(60, 90, 120))
    parser.add_argument("--Ins_size", type=tuple, default=(120, ))
    parser.add_argument("--activation", type=str, default='tanh')
    parser.add_argument("--Tau0", type=tuple, default=(1, 6, 4))
    parser.add_argument("--Tau1", type=tuple, default=(1.2, 4.0, 8.0))
    parser.add_argument("--Tau2", type=tuple, default=(1., 4.0, 8.0))
    
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    
    parser.set_defaults(short_run=False, )
    
    args = parser.parse_args()


    ########## Logging Config ##########
    print("Wandb configuration started...")

    # setup directory for saving training artefacts
    temporary_dir = tempfile.TemporaryDirectory()
    artefacts_dir = Path(temporary_dir.name) / "training_artefacts"
    os.makedirs(str(artefacts_dir))

    # wandb config
    api_key_file = Path("~/.wandbAPIkey.txt").expanduser().resolve()
    project_name = "MNIST1D"
    group_name = args.group_name

    # login to wandb
    with open(api_key_file, "r") as file:
        api_key = file.read().strip()
    wandb.login(key=api_key)

    # initialize wandb
    wandb.init(project=project_name, group=group_name, config={})

    ########## General Config ##########
    print("General configuration started...")

    data_config = dict(default_data_config)
    general_config = dict(default_general_config)
    train_config = dict(default_train_config)
    model_config = dict(default_model_config)

    # General Config
    general_config["seed"] = args.seed
    general_config["dt"] = args.dt
    general_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    general_config["short_training_run"] = args.short_run
    device = torch.device(general_config["device"])
    print("Torch Device: ", device)

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
    model_config["num_LP_layers"] = args.num_LP_layers
    model_config["num_Ins_layers"] = args.num_Ins_layers
    model_config["LP_size"] = args.LP_size
    model_config["Ins_size"] = args.Ins_size
    model_config["activation"] = args.activation
    model_config["Tau0"] = args.Tau0
    model_config["Tau1"] = np.array(args.Tau1)
    model_config["Tau2"] = np.array(args.Tau2)

    # Training Config
    train_config["num_epochs"] = 1 if general_config["short_training_run"] else args.num_epochs
    train_config["learning_rate"] = args.lr
    train_config["batch_size"] = 20 if general_config["short_training_run"] else args.batch
    
    # Dataset Config
    data_config["duration"] = args.duration
    data_config["pad"] = args.pad

    with open(str(artefacts_dir / "general_config.json"), "w", encoding="utf-8") as f:
        json.dump(general_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "data_config.json"), "w", encoding="utf-8") as f:
        json.dump(data_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    wandb.config.update(
        {
            "general_config": general_config,
            "data_config": data_config,
            "model_config": model_config,
            "train_config": train_config,
        }
    )

    dt = general_config["dt"]
    n_steps = data_config["final_seq_length"]
    answer_steps = int(train_config["answer_period"]/dt)
    pad_steps = int(train_config["pad_period"]/dt)
    data_config["final_seq_length"] = int(data_config["duration"]/dt)

    print("Data, model and training setup started...")
    data = make_dataset(ObjectView(data_config))
    x, y = torch.tensor(data['x'],dtype=torch.float32).unsqueeze(-1), torch.tensor(data['y'])
    x_test, y_test = torch.tensor(data['x_test'],dtype=torch.float32).unsqueeze(-1), torch.tensor(data['y_test'])

    # init network and optimizer
    model = buildMNISTNet(model_config, general_config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.75,
        patience=2
    )

    if general_config["visual_kernel"]:
        kernel_record_pre = extract_kernel(
            model,
            n_steps=int(1.5*data_config["final_seq_length"]),
            layer_idx=model_config["num_LP_layers"]
        )

    beta = torch.zeros(n_steps)
    beta[-answer_steps:]=1.
    beta /= beta.sum()
    beta = beta.to(model.device)

    error_record = []
    Cum_errors = 0.5

    print("Training started...")
    with torch.no_grad():
        error_record, loss_record = [], []
        for epoch in range(train_config["num_epochs"]):
            Cum_errors =0.
            permu = np.random.permutation(4000).reshape(-1, train_config["batch_size"])
            pbar = tqdm(
                permu,
                total=permu.shape[0],
            )
            for idx in pbar:
                total_error = train_batch_delay(model, optimizer, x[idx].to(device),
                                                y[idx].to(device), answer_steps, pad_steps, beta)
                Cum_errors += total_error

            test_acc, test_loss = test(model, x_test.to(device), y_test.to(device),
                                    answer_steps, pad_steps, beta)
            scheduler.step(test_loss)
            error_record.append(Cum_errors)   
            print(
                f"Epoch: {epoch+1}, "
                f'Train loss: {Cum_errors:.4f},'
                f'Test Loss: {test_loss:.4f},'
                f'Test Accuracy: {test_acc:.4f},'
            )
        
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state_dict = model.state_dict().copy()

                torch.save(
                    best_model_state_dict, str("../models/new_exp/neuronio_best_model_forget_%r_rest_%r_nummem_%d_%d.pt"%(args.forget_gate, args.rest_start, args.num_memory, args.seed))
                )
            
            # Log statistics
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": Cum_errors / train_config["batches_per_epoch"],
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )

    # Free up memory
    gc.collect()

    # save model for later
    torch.save(
        best_model_state_dict, str(artefacts_dir / "neuronio_best_model_state.pt")
    )
    
    if args.save_model:
        torch.save(
            model.state_dict(), str("../checkpoints/MNIST/parity_best_model_forget_%r_N_%d_nummem_%d_%d.pt"%(args.forget_gate, Ns[0], args.num_memory, args.seed))
        )
    
    # save artefacts to wandb
    wandb.save(
        str(artefacts_dir) + "/*", base_path=str(temporary_dir.name), policy="now"
    )
    wandb.finish()  # finish wandb run
    temporary_dir.cleanup()

    ########## FINISHED ##########
    print("Finished")
