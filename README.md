# MidiBERT-Piano miditok extention

**Jan Jeschke**</br>
**Warsaw University of Technology**

This project extends the capabilities of MidiBERT-Piano by combining it with the miditok package, which provides a lot of easily configurable tokenizations.

Repository heavily based on [MidiBERT-Piano](https://github.com/wazenmai/MIDI-BERT).

To understand the code I recommend reading [MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding](https://arxiv.org/abs/2107.05223).

## Introduction

With this repository, you can:

* pre-train a MidiBERT-Piano with your customized pre-trained dataset
* fine-tune & evaluate on 4 downstream tasks

All the datasets employed in this work are publicly available.

Currently there are 3 implemented tokenizations from miditok package: REMI, TSD and Structured but implementing others should be easy.

## Installation

* Python3
* Install generally used packages for MidiBERT-Piano:

```python
git clone https://github.com/Pinjesz/MIDI-BERT.git
cd MIDI-BERT
pip install -r requirements.txt
```

## A. Prepare Data

Tokenized data for all types of tokenizations is given. It was generated using config in `prepare_data/new/<tokenization_name>/config.py`.

You can also preprocess as below.

### 1. Download Dataset and Preprocess

Save the following dataset in `Dataset/`

* [Pop1K7](https://github.com/YatingMusic/compound-word-transformer)
* [ASAP](https://github.com/fosfrancesco/asap-dataset)
  * Download ASAP dataset from the link
* [POP909](https://github.com/music-x-lab/POP909-Dataset)
  * preprocess to have 865 pieces in qualified 4/4 time signature
  * ```cd preprocess_pop909```
  * ```exploratory.py``` to get pieces qualified in 4/4 time signature and save them at ```qual_pieces.pkl```
  * ```preprocess.py``` to realign and preprocess
  * Special thanks to Shih-Lun (Sean) Wu
* [Pianist8](https://zenodo.org/record/5089279)
  * Step 1: Download Pianist8 dataset from the link
  * Step 2: Run `python3 pianist8.py` to split data by `Dataset/pianist8_(mode).pkl`
* [EMOPIA](https://annahung31.github.io/EMOPIA/)
  * Step 1: Download Emopia dataset from the link
  * Step 2: Run `python3 emopia.py` to split data by `Dataset/emopia_(mode).pkl`

### 2. Prepare Dictionary

Customize the config dictionary in `prepare_data/new/<tokenization_name>/config.py` and then run:

```sh
sh scripts/<tokenization_name>/prepare_data.sh
```

## B. Pre-train a MidiBERT-Piano

```sh
sh scripts/<tokenization_name>/pretrain.sh
```

In the file you can specify the arguments, which are:

* Required:
  * tokenization - name of the tokenization (remi, tsd, structured)
* Optional
  * name - experiment name, defines folder name where results will be saved: `result/<tokenization_name>/pretrain/<name>`, default: default
  * max_seq_len - all input token sequences are padded to this length, default: 512
  * hs - hidden state, default: 768
  * mask_percent - fraction of tokens masked during training, default: 0.15
  * datasets - list of datasets used in training (pop909, composer, pop1k7, ASAP, emopia), default: pop909, composer, pop1k7, ASAP, emopia
  * num_workers - number of dataloader workers, default: 5
  * batch_size - size of the batch, default: 12
  * epochs - max number of training epochs, default: 500
  * lr - initial learning rate, default: 2e-5
  * cpu - if present gpu will not be used
  * cuda_devices - indices of gpu devices, default: 0, 1, 2, 3
  * use_wandb - if present will try to log training data to Weight&Biases
  * project - name of wandb project, default: wimu

## C. Fine-tune & Evaluate on Downstream Tasks

### 1. Fine-tuning

```sh
sh scripts/<tokenization_name>/finetune.sh
```

In the file you can specify the arguments, which are:

* Required:
  * tokenization - name of the tokenization (remi, tsd, structured)
  * task - name of the task (melody, velocity, composer, emotion)
* Optional
  * name - must be the same as used in pretrain, results will be saved in: `result/<tokenization_name>/finetune/<task>_<name>`
  * max_seq_len - must be the same as used in pretrain
  * hs - must be the same as used in pretrain
  * index_layer - number of layers added, default: 12
  * num_workers - number of dataloader workers, default: 5
  * batch_size - size of the batch, default: 12
  * epochs - max number of training epochs, default: 10
  * lr - initial learning rate, default: 2e-5
  * cpu - if present gpu will not be used
  * cuda_devices - indices of gpu devices, default: 0, 1, 2, 3
  * use_wandb - if present will try to log training data to Weight&Biases
  * project - name of wandb project, default: wimu

### 2. Evaluation

```sh
sh scripts/<tokenization_name>/eval.sh
```

In the file you can specify the arguments, which are:

* Required:
  * tokenization - name of the tokenization (remi, tsd, structured)
  * task - name of the task (melody, velocity, composer, emotion)
* Optional
  * name - must be the same as used in finetune, results will be saved in: `result/<tokenization_name>/eval/<name>`
  * max_seq_len - must be the same as used in finetune
  * hs - must be the same as used in finetune
  * index_layer - must be the same as used in finetune
  * num_workers - number of dataloader workers, default: 5
  * batch_size - size of the batch, default: 12
  * lr - initial learning rate, default: 2e-5
  * cpu - if present gpu will not be used
  * cuda_devices - indices of gpu devices, default: 0, 1, 2, 3
  * use_wandb - if present will try to log training data to Weight&Biases
  * project - name of wandb project, default: wimu
