<h1 align="center">ChemAligner-T5</h1>
<p align="center"><a href="#abstract">📝 Paper</a> | <a href="#3-benchmark-datasets">🤗 Benchmark datasets</a> | <a href="https://huggingface.co/collections/Neeze/chemaligner-t5">🚩 Checkpoints</a> | <a href="https://huggingface.co/collections/Neeze/chemaligner-t5">⚙️ Application</a> | <a href="#citation">📚 Cite our paper!</a></p>

The official implementation of manuscript **"ChemAligner-T5: A Unified Text-to-Molecule Model via Representation Alignment"**

## Abstract
> <place_holder for abstract>


## News
- `2025.11.20`: Init source code

## How to use

### 1. Environment preparation
Create an environment using Miniconda or Conda:
```zsh
conda create -n MolLingual python=3.10
conda activate MolLingual
```

After cloning the repo, run the following command to install required packages:
```zsh
# installing pytorch, recommend vervion 2.1.2 or above, you should change cuda version based on your GPU devices
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# installing additional packages
pip install -r requirements.txt

# install additional packages for Torch Geometric, cuda version should match with torch's cuda version
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

### 2. Pretrained models
We use these pretrained models for fine-tuning:

- BioT5+: [HuggingFace](https://huggingface.co/collections/QizhiPei/biot5)
- SwinOCSR: [Kaggle](https://www.kaggle.com/datasets/gogogogo11/moedel)
- SciBERT: [HuggingFace](https://huggingface.co/allenai/scibert_scivocab_uncased)
- GIN-MoMu: [GitHub](https://github.com/ddz16/MoMu)

Except for BioT5 and SciBERT which are automatically downloaded when you start training or evaluating, you need to prepare SwinOCSR and GIN-MoMu's checkpoint from the above link, then put it into `weights/`.

### 3. Benchmark datasets
- LPM-24: [HuggingFace](https://huggingface.co/datasets/duongttr/LPM-24-extend)
- LPM-24-Extra: [HuggingFace](https://huggingface.co/datasets/Neeze/LPM-24-extra-extend)
- CheBI-20: [HuggingFace](https://huggingface.co/datasets/duongttr/chebi-20-new)

Because the datasets are automatically downloaded from HuggingFace, please send access request and login by following command:
```zsh
huggingface-cli login --token '<your_hf_token>'
```

### 3. Preprocess data

Preprocessing datasets

```zsh
python preprocess_data.py --output_dir data/LPM-24-extra-extend \
                          --hf_repo_id Neeze/LPM-24-extra-extend \
                          --num_proc 20 \
                          --hf_token <your_hf_token>
```

### 3. Training model

#### LPM-24 dataset:

SFT BioT5+ scripts

```zsh
python train_lang2mol_base.py --epochs 10 --batch_size 8 \
                --grad_accum 32 --warmup_ratio 0.05 --lr 5e-5 --num_devices 4 \
                --dataset_name lpm-24-extra --model_config src/configs/config_biot5p_base_lpm24_lang2mol_train.yaml --output_folder checkpoints/biot5p_base/SFTBioT5PlusBase --cuda
```


SFT BioT5+ with Contrastive scripts

```zsh
python train_lang2mol_contrastive.py --epochs 10 --batch_size 8 \
                --grad_accum 32 --warmup_ratio 0.05 --lr 5e-5 --num_devices 4 \
                --dataset_name lpm-24-extra --model_config src/configs/config_biot5p_base_contrastive_lpm24_lang2mol_train.yaml --output_folder checkpoints/biot5p_base/SFTBioT5plusBaseContrastive --cuda
```

SFT BioT5+ with Multimodal scripts

```zsh
python train_lang2mol_multimodal.py --epochs 10 --batch_size 8 \
                --grad_accum 32 --warmup_ratio 0.05 --lr 5e-5 --num_devices 4 \
                --dataset_name lpm-24-extra --model_config src/configs/config_biot5p_base_multimodal_lpm24_lang2mol_train.yaml --output_folder checkpoints/biot5p_base/SFTBioT5PlusBaseMutiModal --cuda
```

#### CheBI-20 dataset:
```zsh
python train.py --epochs 50 --batch_size 8 \
                --grad_accum 32 --warmup_ratio 0.04 --lr 1e-4 --num_devices 4 \
                --dataset_name chebi-20 --model_config src/configs/config_chebi20_train.yaml \ 
                --cuda
```

### 4. Evaluating model
#### Evaluate on LPM-24
```zsh
python eval_lang2mol.py --dataset_name lpm-24-eval \
               --model_config src/configs/config_lpm24_lang2mol_train.yaml \
               --output_csv results/results.csv \
               --checkpoint_path path_to_checkpoint.ckpt \
               --cuda
```

#### Evaluate on CheBI-20
```zsh
python eval.py --dataset_name chebi-20 \
               --model_config src/configs/config_chebi20_train.yaml \
               --checkpoint_path path/to/ckpt \
               --cuda
```

#### Push to hub

```zsh
python push_to_hub.py --model_name biot5-plus-base-sft \
                      --ckpt_path checkpoints/biot5p_base/SFTBioT5PlusBase/ckpt_epoch=9_eval_loss=0.22033654153347015.ckpt \
                      --hf_token <your_hf_token>
```

### 5. Application
#### Start the app
You can interact with the model through a user interface by running the following command:

```zsh
python app.py
```

## Citation
If you are interested in my paper, please cite:
```
<place_holder>
```
