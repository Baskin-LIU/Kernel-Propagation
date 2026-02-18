import os
import json
import numpy as np
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from torchaudio.datasets import SPEECHCOMMANDS
#from torchcodec.decoders import AudioDecoder

RAW_ROOT = Path("../SpeechCommands/speech_commands_v0.02")
OUT_ROOT = Path("../SpeechCommands/Mel_npy")

LOG_EPS = 1e-6

def load_split_list(split_name):
    """
    Load official split file and return a set of relative wav paths.
    """
    if split_name == "training":
        val_list = load_split_list("validation")
        test_list = load_split_list("testing")
        all_wavs = set(
            str(p.relative_to(RAW_ROOT)).replace("\\", "/")
            for p in RAW_ROOT.rglob("*.wav")
        )
        return all_wavs - val_list - test_list

    split_file = RAW_ROOT / f"{split_name}_list.txt"
    with open(split_file) as f:
        return set(line.strip() for line in f)


# def process_split(split_name, rewrite=False, n_mels=48, 
#                       n_frames=250, Target_len=16000):
#     print(f"\nProcessing {split_name}")

#     split_files = load_split_list(split_name)
#     out_dir = OUT_ROOT / split_name
#     out_dir.mkdir(parents=True, exist_ok=True)

#     metadata = []
#     label_counts = {}

#     for rel_path in tqdm(sorted(split_files)):
#         wav_path = RAW_ROOT / rel_path

#         label = wav_path.parent.name

#         # Skip background noise folder if desired
#         if label == "_background_noise_":
#             continue

#         # Create label directory
#         label_dir = out_dir / label
#         label_dir.mkdir(exist_ok=True)

#         # Save with original filename but .pt
#         pt_name = wav_path.stem + ".pt"
#         pt_path = label_dir / pt_name

#         metadata.append({
#             "wav_path": str(wav_path),
#             "pt_path": str(pt_path),
#             "label": label,
#             "n_frames": n_frames,
#             "n_mels": n_mels,
#         })

#         label_counts[label] = label_counts.get(label, 0) + 1
        
#         if os.path.isfile(pt_path) and not rewrite:
#             continue
        
#         #waveform, sr = torchaudio.load(wav_path)
#         decoder = AudioDecoder(wav_path)
#         waveform = decoder.get_all_samples().data
#         T = waveform.size(1)
#         if T < Target_len:
#             # pad at the end
#             pad_amount = Target_len - T
#             waveform = F.pad(waveform, (pad_amount, 0))

#         mel = mel_transform(waveform)[0, :, :-1] # (n_mels, T)
#         assert mel.shape[1]==250
#         mel = torch.log(mel + LOG_EPS)
#         torch.save(mel, pt_path)
        

#     # Save metadata
#     pd.DataFrame(metadata).to_csv(out_dir / "metadata.csv", index=False)

#     # Save label statistics
#     with open(out_dir / "label_stats.json", "w") as f:
#         json.dump(label_counts, f, indent=4)

#     print("Label distribution:")
#     print(label_counts)


def process_split_sharded(
        split_name,
        rewrite=False,
        n_mels=48,
        n_frames=250,
        Target_len=16000,
        shard_size=11100,
        ):

    print(f"\nProcessing {split_name}")

    split_files = load_split_list(split_name)
    out_dir = OUT_ROOT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    label_counts = {}
    label_to_idx = {}
    next_label_idx = 0

    shard_mels = []
    shard_labels = []

    shard_id = 0
    sample_index = 0
    shard_records = []

    def flush_shard():
        nonlocal shard_id, shard_mels, shard_labels

        if len(shard_mels) == 0:
            return

        mel_array = np.stack(shard_mels)
        label_array = np.array(shard_labels)

        mel_path = out_dir / f"shard_{shard_id:03d}_mel.npy"
        label_path = out_dir / f"shard_{shard_id:03d}_labels.npy"

        np.save(mel_path, mel_array)
        np.save(label_path, label_array)

        shard_records.append({
            "shard_id": shard_id,
            "num_samples": len(label_array),
            "mel_file": mel_path.name,
            "label_file": label_path.name
        })

        print(f"Saved shard {shard_id} with {len(label_array)} samples")

        shard_mels = []
        shard_labels = []
        shard_id += 1

    # -------- iterate dataset --------
    for rel_path in tqdm(sorted(split_files)):

        wav_path = RAW_ROOT / rel_path
        label = wav_path.parent.name

        if label == "_background_noise_":
            continue

        if label not in label_to_idx:
            label_to_idx[label] = next_label_idx
            next_label_idx += 1

        label_idx = label_to_idx[label]

        waveform,_ = torchaudio.load(wav_path)

        T = waveform.size(1)
        if T < Target_len:
            pad_len = Target_len - T
            pre = pad_len//2
            post = pad_len - pre
            waveform = F.pad(waveform, (pre, post))
        else:
            waveform = waveform[:, :Target_len]

        mel = mel_transform(waveform)[0, :, :-1]
        mel = torch.log(mel + LOG_EPS)

        mel_np = mel.numpy().astype(np.float32)

        shard_mels.append(mel_np)
        shard_labels.append(label_idx)

        metadata.append({
            "index": sample_index,
            "wav_path": str(wav_path),
            "label_str": label,
            "label_idx": label_idx,
            "shard_id": shard_id
        })

        label_counts[label] = label_counts.get(label, 0) + 1
        sample_index += 1

        if len(shard_mels) >= shard_size:
            flush_shard()

    flush_shard()

    # -------- save metadata --------
    pd.DataFrame(metadata).to_csv(out_dir / "metadata.csv", index=False)

    with open(out_dir / "label_stats.json", "w") as f:
        json.dump({
            "label_counts": label_counts,
            "label_to_idx": label_to_idx
        }, f, indent=4)

    with open(out_dir / "shard_index.json", "w") as f:
        json.dump(shard_records, f, indent=4)

    print("\nFinished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ### General config
    parser.add_argument("--rewrite", dest="rewrite", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--duration", type=float, default=1.)
    
    parser.set_defaults(rewrite=False)
    
    args = parser.parse_args()
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=400,
        win_length=400,
        hop_length=args.sample_rate//args.n_frames,
        n_mels=args.n_mels,
        center=True
    )
    
    if not os.path.isdir(RAW_ROOT):
        SC = SPEECHCOMMANDS("../", download=True)
        del SC

    for split in ["training", "validation", "testing"]:
        process_split_sharded(split, rewrite=args.rewrite, n_mels=args.n_mels, 
                      n_frames=args.n_frames, Target_len=int(args.duration*args.sample_rate))
