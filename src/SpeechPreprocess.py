import os
import json
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torchaudio.datasets import SPEECHCOMMANDS

RAW_ROOT = Path("../SpeechCommands/speech_commands_v0.02")
OUT_ROOT = Path("../SpeechCommands/Mel")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=64,
    n_mels=54,
    center=True
)
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


def process_split(split_name):
    print(f"\nProcessing {split_name}")

    split_files = load_split_list(split_name)
    out_dir = OUT_ROOT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    label_counts = {}

    for rel_path in tqdm(sorted(split_files)):
        wav_path = RAW_ROOT / rel_path

        label = wav_path.parent.name

        # Skip background noise folder if desired
        if label == "_background_noise_":
            continue

        waveform, sr = torchaudio.load(wav_path)
        T = waveform.size(1)
        TARGET_LEN = 16000 
        if T < TARGET_LEN:
            # pad at the end
            pad_amount = TARGET_LEN - T
            waveform = F.pad(waveform, (pad_amount, 0))

        mel = mel_transform(waveform)[0, :, :-1] # (n_mels, T)
        assert mel.shape[1]==250
        mel = torch.log(mel + LOG_EPS)

        # Create label directory
        label_dir = out_dir / label
        label_dir.mkdir(exist_ok=True)

        # Save with original filename but .pt
        pt_name = wav_path.stem + ".pt"
        pt_path = label_dir / pt_name
        torch.save(mel, pt_path)

        metadata.append({
            "wav_path": str(wav_path),
            "pt_path": str(pt_path),
            "label": label,
            "n_frames": mel.shape[1]
        })

        label_counts[label] = label_counts.get(label, 0) + 1

    # Save metadata
    pd.DataFrame(metadata).to_csv(out_dir / "metadata.csv", index=False)

    # Save label statistics
    with open(out_dir / "label_stats.json", "w") as f:
        json.dump(label_counts, f, indent=4)

    print("Label distribution:")
    print(label_counts)


if __name__ == "__main__":
    if not os.path.isdir(RAW_ROOT):
        SC = SPEECHCOMMANDS("../", download=True)
        del SC

    for split in ["training", "validation", "testing"]:
        process_split(split)
