# [Moreh] Running on HAC VM - Moreh AI Framework

## Prepare
### Data
Download and extract the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
The dataset directory should be named `LJSpeech-1.1`.

### Environment
First, create conda environment:
```bash
conda create -n fastspeech python=3.8
conda activate fastspeech
```

#### On HAC VM
Install `torch` first:
```bash
conda install -y torchvision torchaudio numpy protobuf==3.13.0 pytorch==1.7.1 cpuonly -c pytorch
```
Then force update Moreh with latest version (`22.9.0` at the moment this document is written):
```bash
update-moreh --force --target 22.9.0
```
Comment out `torch` in `requirements.txt`, and then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

#### On A100 VM
Comment out `torch` in `requirements.txt`. Then, install the dependencies:
```bash
pip install -r requirements.txt
```

Then, install `torch` with correct CUDA:
```bash
# torch 1.7.1
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# or torch 1.12.1
conda install pytorch torchvision torchaudio cudatoolkit=1.13 -c pytorch
```

### Code
Clone the repo:
```bash
git clone https://github.com/xcmyz/FastSpeech
cd FastSpeech
```

## Run

### Prepare dataset
First, copy the `LJSpeech-1.1` dataset folder to `./data` (symlink also works).
Then, unzip `alignments.zip`:
```bash
unzip ./alignments.zip
```
Download the pretrained waveglow model in [this ggdrive link](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing)
and put it in `./waveglow/pretrained_model/`. Rename the pretrained model to `waveglow_256channels.pt`.
Run the preprocess script:
```bash
python preprocess.py
```

### Train
Edit `hparams.py` and adjust `epochs` & `log_step` to appropriate values.
Some recommendations are `epochs=20`, `log_step=20`.
Then:
```bash
python train.py
```

### Evaluation
After training is done, checkpoints will be saved in `./model_new` in format of
`checkpoint_{number of steps}.pth.tar` (e.g. `checkpoint_3000.pth.tar`)

To evaluation the checkpoint at 3000 steps:
```
python eval.py --step 3000
```

# Original README content:
---
# FastSpeech-Pytorch -
The Implementation of FastSpeech Based on Pytorch.

## Update (2020/07/20)
1. Optimize the training process.
2. Optimize the implementation of length regulator.
3. Use the same hyper parameter as FastSpeech2.
4. **The measures of the 1, 2 and 3 make the training process 3 times faster than before.**
5. **Better speech quality.**

## Model
<div style="text-align: center">
    <img src="img/fastspeech_structure.png" style="max-width:100%;">
</div>

## My Blog
- [FastSpeech Reading Notes](https://zhuanlan.zhihu.com/p/67325775)
- [Details and Rethinking of this Implementation](https://zhuanlan.zhihu.com/p/67939482)

## Prepare Dataset
1. Download and extract [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Put LJSpeech dataset in `data`.
3. Unzip `alignments.zip`.
4. Put [Nvidia pretrained waveglow model](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing) in the `waveglow/pretrained_model` and rename as `waveglow_256channels.pt`;
5. Run `python3 preprocess.py`.

## Training
Run `python3 train.py`.

## Evaluation
Run `python3 eval.py`.

## Notes
- In the paper of FastSpeech, authors use pre-trained Transformer-TTS model to provide the target of alignment. I didn't have a well-trained Transformer-TTS model so I use Tacotron2 instead.
- I use the same hyper-parameter as [FastSpeech2](https://arxiv.org/abs/2006.04558).
- The examples of audio are in `sample`.
- [pretrained model](https://drive.google.com/file/d/1vMrKtbjPj9u_o3Y-8prE6hHCc6Yj4Nqk/view?usp=sharing).

## Reference

### Repository
- [The Implementation of Tacotron Based on Tensorflow](https://github.com/keithito/tacotron)
- [The Implementation of Transformer Based on Pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [The Implementation of Transformer-TTS Based on Pytorch](https://github.com/xcmyz/Transformer-TTS)
- [The Implementation of Tacotron2 Based on Pytorch](https://github.com/NVIDIA/tacotron2)
- [The Implementation of FastSpeech2 Based on Pytorch](https://github.com/ming024/FastSpeech2)

### Paper
- [Tacotron2](https://arxiv.org/abs/1712.05884)
- [Transformer](https://arxiv.org/abs/1706.03762)
- [FastSpeech](https://arxiv.org/abs/1905.09263)
- [FastSpeech2](https://arxiv.org/abs/2006.04558)
