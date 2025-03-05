#!/usr/bin/env python3
import argparse
import glob
import librosa
import os
import torch
import yaml

import utils
from dataset import MelSpectrogram
from model import Model
from model import remove_parametrizations


def main(args):
    device = torch.device("cuda")
    with open(os.path.join(os.path.dirname(args.checkpoint_file), "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    model = Model(cfg["model"]).to(device)
    print(f"Loading {args.checkpoint_file}")
    model.load_state_dict(torch.load(args.checkpoint_file, map_location=device)["model"])

    generator = model.vae.generative_model
    generator.apply(remove_parametrizations)
    generator.eval()

    melspectrogram = MelSpectrogram(cfg["data"]["sample_rate"], **cfg["data"]["stft"], **cfg["data"]["msg"])

    filelist = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    for wavfile in filelist:
        wav, sr = utils.load_wav(wavfile)
        wav = librosa.util.normalize(wav) * 0.95
        msg = melspectrogram(torch.tensor(wav, dtype=torch.float)[None]).to(device)

        with torch.no_grad():
            output = generator(msg).mu.flatten().cpu().numpy()

        output_file = os.path.join(args.output_dir, os.path.basename(wavfile).replace(".wav", "_v2coder.wav"))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        utils.save_wav(output, output_file, cfg["data"]["sample_rate"])
        print(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint_file")
    parser.add_argument("-i", "--input_dir")
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()

    seed = 12345
    torch.cuda.manual_seed(seed)
    main(args)
