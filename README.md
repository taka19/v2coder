# V2Coder: A Non-Autoregressive Vocoder Based on Hierarchical Variational Autoencoders
**Takato Fujimoto, Kei Hashimoto, Yoshihiko Nankaku, Keiichi Tokuda**

[PyTorch](https://pytorch.org/) implementation of V2Coder.
Visit our [website](https://www.sp.nitech.ac.jp/~taka19/demos/v2coder-demo/) for audio samples.

## Setup
1. Clone our repo
    ```
    git clone https://github.com/taka19/v2coder.git
    cd v2coder
    ```
2. Install requirements
    ```
    pip3 install -r requirements.txt
    ```
## Training
```
CORPUS=<path to corpus> python train.py --config_file config.yaml --checkpoint_dir checkpoints
```

## Inference
```
python inference.py --checkpoint_file <checkpoint file> --input_dir <directory containing *.wav files> --output_dir <output directory>
```