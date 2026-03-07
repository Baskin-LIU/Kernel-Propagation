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
from utilsSpeech import *

import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--group_name", type=str, default="test")
    parser.add_argument("--save_local", dest="save_local", action="store_true")
    ### General config
    parser.add_argument("--short_run", dest="short_run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=2.) #ms
    parser.add_argument("--visual_kernel", dest="visual_kernel", action="store_true")
    #parser.add_argument("--machine", type=str, default="MLcloud")
    
    ### Training config
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--answer_t", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--method", type=str, default='KP')
    parser.add_argument("--update_interval", type=int, default=-1)
    parser.add_argument("--error_steps", type=int, default=20)
    parser.add_argument("--cont_train", dest="cont_train", action="store_true")

    ### Data config
    parser.add_argument("--task", type=str, default="Cmd20")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--mask", type=int, default=0)
    parser.add_argument("--mask_fre", type=int, default=4)
    parser.add_argument("--max_warp", type=int, default=25)
    parser.add_argument("--max_shift", type=int, default=25)
    parser.add_argument("--mel", type=int, default=80)
    parser.add_argument("--weighted_sampler", dest="weighted_sampler", action="store_true")
    
    ### Model config
    parser.add_argument("--activation", type=str, default='tanh')
    parser.add_argument("--num_LP_layers", type=int, default=4)
    parser.add_argument("--num_Ins_layers", type=int, default=1)
    parser.add_argument("--LP_size", type=int, nargs="+",
        help="Hidden Low-pass layer sizes", default=[240, 300, 300, 360],)
    parser.add_argument("--Ins_size", type=int, nargs="+", default=[360, ],)
    parser.add_argument("--Tau0", type=int, nargs=3, default=[2, 40, 6],)
    parser.add_argument("--Tau1", type=float, nargs="+", default=[3, 12, 24],)
    parser.add_argument("--Tau2", type=float, nargs="+", default=[4, 16, 36],)
    parser.add_argument("--Tau3", type=float, nargs="+", default=[5, 14],)
    parser.add_argument("--Tau4", type=float, nargs="+", default=[6, 15],)
    parser.add_argument("--TauL", type=float, nargs="+", default=[50, 400],) #Last LP layer
    
    parser.set_defaults(short_run=False, visual_kernel=False, save_local=True, weighted_sampler=False, cont_train=False)
    
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
    
    for i in range(model_config["num_LP_layers"]-1):
        model_config["Tau%d"%i] =getattr(args, "Tau%d"%i)
    model_config["Tau%d"%(i+1)] =getattr(args, "TauL")

    # Training Config
    train_config["num_epochs"] = 1 if general_config["short_training_run"] else args.num_epochs
    train_config["learning_rate"] = args.lr
    train_config["batch_size"] = args.batch
    train_config["num_workers"] = args.workers
    train_config["method"] = args.method
    train_config["update_interval"] = args.update_interval
    train_config["weighted_sampler"] = args.weighted_sampler
    train_config["error_steps"] = args.error_steps
    
    # Dataset Config
    dt = general_config["dt"]
    answer_steps = int(model_config["answer_period"]/dt)

    print("Data, model and training setup started...")
    # Create training and testing split of the data. We do not use validation in this tutorial.
    data_config["mask_width"] = args.mask
    data_config["mask_width_fre"] = args.mask_fre
    data_config["max_warp"] = args.max_warp
    data_config["max_shift"] = args.max_shift
    data_config["n_steps"] = int(data_config["duration"]/dt)
    data_config["task"] = args.task
    
    if data_config["task"]=="Cmd20":
        data_config["n_class"] = len(COMMAND20)
    elif data_config["task"]=="Full":
        data_config["n_class"] = len(LABELS)
    elif data_config["task"]=="WD1600":
        data_config["n_class"] = len(WORDS1600)
    else:
        raise NotImplementedError
    n_class = data_config["n_class"]
    n_steps = data_config["n_steps"]
    answer_steps = int(model_config["answer_period"]/dt)
    if args.mel == 80:
        rootdir = Path('..') / "SpeechCommands" / "Mel_80"
    elif args.mel == 64:
        rootdir = Path('..') / "SpeechCommands" / "Mel_npy"
    else:
        rootdir = Path('..') / "SpeechCommands" / "Mel"
    
    train_set = ShardedMelDataset(rootdir / 'training', task=data_config["task"])
    val_set = ShardedMelDataset(rootdir / 'validation', task=data_config["task"])
    
    mel_sample, label = val_set[0]
    data_config['n_mels'] = mel_sample.shape[0]
    ori_len = mel_sample.shape[1]
    data_config['hop_length'] = (16000/ori_len)
    
    if device == "cuda":
        num_workers = train_config['num_workers']
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    if train_config["weighted_sampler"]:
        with open(rootdir / "training"/ "label_stats.json") as f:
            label_counts = json.load(f)["label_counts"]
        sampler = make_weighted_sampler(train_set, label_counts)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_config['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=CollateMel(n_steps, n_class, training=True,
                              ori_len = ori_len, 
                              mask_width=data_config["mask_width"], 
                              mask_width_fre=data_config["mask_width_fre"],
                              max_warp = data_config["max_warp"],
                              max_shift = data_config["max_shift"],
                              ),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        collate_fn=CollateMel(n_steps, n_class, training=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    train_config["batches_per_epoch"] = len(train_loader)
    
    model_config['n_out'] = data_config['n_class']
    model_config['n_in'] = data_config['n_mels']
    
    beta = torch.zeros(n_steps).to(device)
    beta[-answer_steps:]=1.
    beta /= beta.sum()

    # Init network and optimizer
    if args.method=='BPTT':
        model = buildNetCompare(model_config, general_config, neurontype=args.method).to(device)
        train_fn = train_batch_BPTT
    else:
        if args.update_interval==-1:
            train_fn = train_batch_delay
        else:
            train_fn = train_batch_periodic(args.update_interval) #Not Implemented Yet
        if args.method=='KP':
            model = buildKPNet(model_config, general_config).to(device)
        elif args.method=='GLE':
            model = buildNetCompare(model_config, general_config, neurontype=args.method).to(device)
        elif args.method=='RFLO':
            model = buildNetCompare(model_config, general_config, neurontype=args.method).to(device)
        else:
            raise NotImplementedError

    model = torch.compile(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=(0.9, 0.999)
    )
    
    train_config["factor"] = 0.5 if args.method=='BPTT' else 0.6
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_config["factor"],
        patience=2
    )

    if args.cont_train: #load midway model
        ckp = torch.load(Path("..") / "midway_models" / (args.group_name+f"{args.seed}.pt"))
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
    
    beta = torch.zeros(n_steps).to(model.device)
    beta[-answer_steps:]=1.
    beta /= beta.sum()
    
    best_val_acc = 0.

    ########## Wandb Logging Config ##########
    print("Wandb configuration started...")

    # setup directory for saving training artefacts
    temporary_dir = tempfile.TemporaryDirectory()
    artefacts_dir = Path(temporary_dir.name) / "training_artefacts"
    os.makedirs(str(artefacts_dir))

    # wandb config
    api_key_file = Path("~/.wandbAPIkey.txt").expanduser().resolve()
    project_name = "KPSpeech"
    group_name = args.task + args.method + args.group_name

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

    ######## Training #############################
    print("Training started...")
    
    loss_record = []
    for epoch in range(train_config["num_epochs"]):
        Cum_loss=0.
        pbar = tqdm(
            enumerate(train_loader, 0),
            total=train_config["batches_per_epoch"],
            disable=not general_config["verbose"],
        )
        for batch_idx, (x, target) in pbar:
            if args.error_steps==-1: #All steps in answering period
                beta_temp = beta
            else: #Random selected steps in answering period
                beta_temp = torch.zeros(n_steps).to(device)
                select_steps = np.random.choice(answer_steps, args.error_steps, replace=False)
                beta_temp[select_steps+n_steps-answer_steps] = 1.
                beta_temp /= beta_temp.sum()
            total_loss = train_fn(model, optimizer, x.to(device),
                                        target.to(device), answer_steps, beta_temp)
            Cum_loss += total_loss
        Cum_loss /= len(train_loader.dataset)

        val_acc, val_loss = test(model, val_loader, answer_steps, beta)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state_dict = model.state_dict().copy()
        
        if args.method=='KP':
            checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }
            torch.save(checkpoint, Path("..") / "midway_models" / (args.group_name+f"{args.seed}.pt"))

        print(
            f"Epoch: {epoch+1}, "
            f'Train Loss: {Cum_loss:.4f},'
            f'Val Loss: {val_loss:.4f},'
            f'Val Acc: {val_acc:.1f},'
            f'Best Acc: {best_val_acc:.1f},'
        )
        
        # Log statistics
        wandb.log(
            {
                "epoch": epoch+1,
                "train_loss": Cum_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        pbar.close()
            
    wandb.log({"Best Val Acc": best_val_acc})

    del train_loader
    del val_loader
    # Free up memory
    gc.collect()

    print("Evaluation on best model...")
    test_set = ShardedMelDataset(rootdir / 'testing', task=data_config["task"])
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        collate_fn=CollateMel(n_steps, n_class, training=False),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    model.load_state_dict(best_model_state_dict)
    test_acc, test_loss = test(model, test_loader, answer_steps, beta)
    print(
                f'Test Loss: {test_loss:.4f},'
                f'Test Acc: {test_acc:.1f},'
                f'Best Val Acc: {best_val_acc:.1f},'
        )
    wandb.log({"test_loss": test_loss, "test_acc": test_acc,})

    # save model for later
    torch.save(
        best_model_state_dict, str(artefacts_dir / "best_model_state.pt")
    )
    
    if args.save_local and args.method=='KP':
        print("Saved local at saved_models \ Speech%.1f"%best_val_acc)
        shutil.copytree(artefacts_dir, Path("..") / "saved_models" / f"Speech{best_val_acc:.1f}", dirs_exist_ok=True)
    
    # save artefacts to wandb
    wandb.save(
        str(artefacts_dir) + "/*", base_path=str(temporary_dir.name), policy="now"
    )
    wandb.finish()  # finish wandb run
    temporary_dir.cleanup()

    ########## FINISHED ##########
    print("Finished")
