# V2Coder: A Non-Autoregressive Vocoder Based on Hierarchical Variational Autoencoders
**Takato Fujimoto, Kei Hashimoto, Yoshihiko Nankaku, Keiichi Tokuda**

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
The audio files and the file list are organized as follows:
```
+-- CORPUS_NAME
|   --- training.txt
|   +-- wavs
|       +-- speaker1
|            --- spkr1_file1.wav
|            --- spkr1_file2.wav
|            ...
|       --- file1.wav
|       --- file2.wav
|       ...
```
The contents of the file list (training.txt) should include entries like:
```
speaker1/spkr1_file1
speaker1/spkr1_file2
...
file1
file2
...
```
To start the training, run the following command:
```
CORPUS=CORPUS_NAME python train.py --config_file config.yaml --checkpoint_dir checkpoints
```

## Inference
```
python inference.py --checkpoint_file checkpoints/latest.ckpt --input_dir test_wavs --output_dir generated_wavs
```