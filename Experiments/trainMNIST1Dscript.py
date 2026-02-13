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
import shutil
import copy
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "../src/")
from Network import *
from utils import *
from plotting import *
from utilsMNIST import *

from mnist1d.data import make_dataset
from mnist1d.utils import ObjectView, set_seed
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--group_name", type=str, default="test")
    parser.add_argument("--save_local", dest="save_local", action="store_true")
    ### General config
    parser.add_argument("--short_run", dest="short_run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.) #ms
    parser.add_argument("--visual_kernel", dest="visual_kernel", action="store_true")
    #parser.add_argument("--machine", type=str, default="MLcloud")
    
    ### Training config
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--answer_t", type=int, default=300)
    parser.add_argument("--method", type=str, default='KP')
    parser.add_argument("--update_interval", type=int, default=-1)

    ### Data config
    parser.add_argument("--prepad", type=int, default=10)
    parser.add_argument("--batch", type=int, default=100)
    
    ### Model config
    parser.add_argument("--activation", type=str, default='tanh')
    parser.add_argument("--num_LP_layers", type=int, default=4)
    parser.add_argument("--num_Ins_layers", type=int, default=1)
    parser.add_argument("--LP_size", type=int, nargs="+",
        help="Hidden Low-pass layer sizes", default=[90, 90, 90, 90],)
    parser.add_argument("--Ins_size", type=int, nargs="+", default=[120, ],)
    parser.add_argument("--Tau0", type=int, nargs=3, default=[1, 10, 6],)
    parser.add_argument("--Tau1", type=int, nargs="+", default=[3, 6],)
    parser.add_argument("--Tau2", type=int, nargs="+", default=[2, 7],)
    parser.add_argument("--Tau3", type=int, nargs="+", default=[1, 8., 12.],)
    
    parser.set_defaults(short_run=False, visual_kernel=False, save_local=True)
    
    args = parser.parse_args()

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
    general_config["visual_kernel"] = args.visual_kernel
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
    model_config["LP_size"] = args.LP_size[:args.num_LP_layers]
    model_config["Ins_size"] = args.Ins_size[:args.num_Ins_layers]
    model_config["activation"] = args.activation
    model_config["answer_period"] = args.answer_t
    
    for i in range(model_config["num_LP_layers"]):
        model_config["Tau%d"%i] =getattr(args, "Tau%d"%i)

    # Training Config
    train_config["num_epochs"] = 1 if general_config["short_training_run"] else args.num_epochs
    train_config["learning_rate"] = args.lr
    train_config["batch_size"] = args.batch
    train_config["method"] = args.method
    train_config["update_interval"] = args.update_interval
    
    # Dataset Config
    data_config["prepad"] = args.prepad

    dt = general_config["dt"]
    n_steps = data_config["final_seq_length"]
    answer_steps = int(model_config["answer_period"]/dt)
    pad_steps = int(data_config["prepad"]/dt)
    data_config["final_seq_length"] = int(data_config["duration"]/dt)

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

    print("Data, model and training setup started...")
    data = make_dataset(ObjectView(data_config))
    x, y = torch.tensor(data['x'],dtype=torch.float32).unsqueeze(-1), torch.tensor(data['y'], dtype=torch.int64)
    x_test, y_test = torch.tensor(data['x_test'],dtype=torch.float32).unsqueeze(-1), torch.tensor(data['y_test'], dtype=torch.int64)

    # init network and optimizer
    if args.method=='BPTT':
        model = buildKPNet(model_config, general_config).to(device)
        train_fn = train_batch_BPTT
    else:
        if args.update_interval==-1:
            train_fn = train_batch_delay
        else:
            train_fn = train_batch_periodic(args.update_interval)
        if args.method=='KP':
            model = buildKPNet(model_config, general_config).to(device)
        elif args.method=='GLE':
            model = buildNetCompare(model_config, general_config, neurontype=args.method).to(device)
        elif args.method=='RFLO':
            model = buildNetCompare(model_config, general_config, neurontype=args.method).to(device)
        else:
            raise NotImplementedError

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

    beta = torch.zeros(n_steps).to(model.device)
    beta[-answer_steps:]=1.
    beta /= beta.sum()

    best_test_acc = 0.

    print("Training started...")
    error_record, loss_record = [], []
    for epoch in range(train_config["num_epochs"]):
        Cum_errors=0.
        permu = np.random.permutation(4000).reshape(-1, train_config["batch_size"])
        pbar = tqdm(
            permu,
            total=permu.shape[0],
        )
        for idx in pbar:
            total_error = train_batch_delay(model, optimizer, x[idx].to(device),
                                            y[idx].to(device), answer_steps, pad_steps, beta)
            Cum_errors += total_error
        Cum_errors /= permu.shape[0]

        test_acc, test_loss = test(model, x_test.to(device), y_test.to(device),
                                answer_steps, pad_steps, beta)
        scheduler.step(test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch: {epoch+1}, "
            f'Train Loss: {Cum_errors:.4f},'
            f'Test Loss: {test_loss:.4f},'
            f'Test Acc: {test_acc:.1f},'
            f'Best Acc: {best_test_acc:.1f},'
        )
        
        # Log statistics
        wandb.log(
            {
                "epoch": epoch+1,
                "train_loss": Cum_errors,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
    wandb.log({"Best Test Acc": best_test_acc})
    # Free up memory
    gc.collect()

    # save model for later
    torch.save(
        best_model_state_dict, str(artefacts_dir / "best_model_state.pt")
    )
    
    if args.save_local:
        print("Saved local at saved_models\MNIST1D%.1f"%best_test_acc)
        shutil.copytree(artefacts_dir, Path("..") / "saved_models" / f"MNIST1D{best_test_acc:.1f}", dirs_exist_ok=True)
    
    # save artefacts to wandb
    wandb.save(
        str(artefacts_dir) + "/*", base_path=str(temporary_dir.name), policy="now"
    )
    wandb.finish()  # finish wandb run
    temporary_dir.cleanup()

    ########## FINISHED ##########
    print("Finished")
