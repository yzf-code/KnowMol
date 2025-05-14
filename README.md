# KnowMol: Advancing Molecular Large Language Models with Multi-Level Chemical Knowledge
Codes for our paper *KnowMol: Advancing Molecular Large Language Models with Multi-Level Chemical Knowledge*



## Overview
<p align="center">
    <a> <img src="assets/static/teaser.png" width="100%"> </a>
</p>
The molecular large language models have garnered widespread attention due to their promising potential on molecular applications. However, current molecular large language models face significant limitations in understanding molecules due to inadequate textual descriptions and suboptimal molecular representation strategies during pretraining. To address these challenges, we introduce KnowMol-100K, a large-scale dataset with 100K fine-grained molecular annotations across multiple levels, bridging the gap between molecules and textual descriptions. Additionally, we propose chemically-informative molecular representation, effectively addressing limitations in existing molecular representation strategies. Building upon these innovations, we develop KnowMol, a state-of-the-art multi-modal molecular large language model. Extensive experiments demonstrate that KnowMol achieves superior performance across molecular understanding and generation tasks.

<!-- ## Architecture
The diagram presented below provides an overview of the architectural design of the InstructMol model, along with its two-stage training paradigm. The example molecule in the figure is Terephthalaldehyde (CID 12173).
<p align="center">
    <a> <img src="pics/overview.png" width="80%"> </a>
</p> -->

## Release
<!-- - [2023/11/27] 🔥 We first release our code (including training and evaluation scripts). -->


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna, LLaVA, Mol-Instructions and GPT-4. The dataset is using MIT license and models trained using the dataset should not be used outside of research purposes.


## Contents
- [Install](#install)
- [Weights](#weights)
- [Dataset](#dataset)
- [Train](#train)
- [Evaluation](#evaluation)

## Install
Mostly refer to LLaVA installation
1. Clone this repository and navigate to project folder

2. Install Package
- If you have any trouble install torch-geometric related packages, please refer to [guide-to-pyg-install](https://github.com/chao1224/GraphMVP#environments) for detailed instructions.
```Shell
conda create -n KnowMol python=3.10 -y
conda activate KnowMol
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install Graph related packages. We use torch-112 with CUDA-11.6, please change accordingly.
pip install -r requirements.txt
```

3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn --no-build-isolation
```


## Weights

### Component Weights Download
Create a folder named `checkpoints` in the root directory of this project. 
```Shell
mkdir checkpoints
cd checkpoints
```
Download the following weights and put them in the `checkpoints` folder.
```Shell
# Under the checkpoints folder
# get the weights for the vicuna model (https://huggingface.co/lmsys/vicuna-7b-v1.3)
ln -s YOUR_PATH_TO_vicuna_v1_3_7b vicuna-v1-3-7b
# get the weights for MoleculeSTM model
mkdir MoleculeSTM
wget https://huggingface.co/chao1224/MoleculeSTM/resolve/main/demo/demo_checkpoints_Graph/molecule_model.pth -P MoleculeSTM
# download the weights for scibert_scivocab_uncased model (https://huggingface.co/allenai/scibert_scivocab_uncased)
ln -s YOUR_PATH_TO_scibert_scivocab_uncased scibert_scivocab_uncased
cd .. # back to the root directory
```
* [Optional] Get graphmvp weights, please refer to [GraphMVP weights download guidance](https://github.com/chao1224/GraphMVP#for-graphmvp-pre-training). 
    ```Shell
    mv YOUR_PATH_TO_graphmvp.pth checkpoints/
    ```


## Dataset
* We have upload our KnowMol-100K dataset on Hugging Face. 
Please see the full dataset in https://huggingface.co/datasets/yzf1102/KnowMol-100K




## Train
KnowMol training consists of two stages:

* **Stage 1: Pretraining.** 
* **Stage 2: Task-specific Instruction Tuning.** The second stage fine-tunes KnowMol for specific downstream tasks, allowing it to effectively interpret and follow human instructions, thereby enhancing the model’s performance across various applications. We also utilize LoRA to improve efficiency.

### Stage 1: Pretraining
See [pretrain.sh](scripts/pretrain.sh) for an example of how to run the pretraining stage.
- `$GRAPH_TOWER` can be chosen from `moleculestm` or `graphmvp`.

### Stage 2: Task-specific Instruction Tuning
You can train each task with specific script. (e.g., [molecule description generation task](scripts/finetune_lora_molcap.sh)).


## Evaluation
See [Evaluation.md](Evaluation.md) for detailed instructions on how to evaluate the model.

<!-- ## Citation
If you find KnowMol useful for your your research and applications, please cite using this BibTeX:
```bibtex

``` -->

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the main base-LLM we used.
- [InstructMol](https://github.com/IDEA-XL/InstructMol): the codebase we built upon.