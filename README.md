<h1 align="center">ChemAligner-T5</h1>
<p align="center"><a href="#abstract">📝 Paper</a> | <a href="#3-benchmark-datasets">🤗 Benchmark datasets</a> | <a href="https://huggingface.co/collections/Neeze/chemaligner-t5">🚩 Checkpoints</a> | <a href="https://huggingface.co/collections/Neeze/chemaligner-t5">⚙️ Application</a> | <a href="#citation">📚 Cite our paper!</a></p>

The official implementation of manuscript **"ChemAligner-T5: A Unified Text-to-Molecule Model via Representation Alignment"**

## Abstract
> Molecular generation from natural language descriptions is becoming an important approach for guided molecule design, as it allows researchers to express chemical objectives directly in textual form. However, string representations such as SMILES and SELFIES reside in embedding spaces that differ significantly from natural language, creating a mismatch that prevents generative models from accurately capturing the intended chemical semantics. This gap raises the question of whether a shared representation space can be constructed in which textual descriptions and molecular strings converge in a controlled manner. Motivated by this gap, we introduce ChemAligner-T5, a BioT5+ base model enhanced with a contrastive learning mechanism to directly align textual and molecular representations. On the L+M-24 test set, ChemAligner-T5 achieves a BLEU score of 69.77\% and a Levenshtein distance of 31.28\%, outperforming MolT5-base and Meditron on both metrics. Visual analysis shows that the model successfully reproduces the structural scaffold and key functional groups of the target molecule. These results highlight the importance of text–molecule representation alignment for the Text2Mol task and strengthen the potential of language models as direct interfaces for molecule design and drug discovery guided by natural-language descriptions.


## News
- `2025.11.20`: Init source code

## How to use

### 1. Environment preparation
Create an environment using Miniconda or Conda:
```zsh
conda create -n ChemAligner python=3.10
conda activate ChemAligner
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
- BioT5+: [HuggingFace](https://huggingface.co/collections/QizhiPei/biot5)

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
                --grad_accum 32 --warmup_ratio 0.00 --lr 5e-4 --num_devices 4 \
                --dataset_name lpm-24-extra --model_config src/configs/config_lpm24_train.yaml --output_folder checkpoints/biot5p_base/SFTBioT5PlusBase --cuda
```


SFT BioT5+ with Contrastive scripts

```zsh
python train_lang2mol_contrastive.py --epochs 10 --batch_size 8 \
                --grad_accum 32 --warmup_ratio 0.00 --lr 5e-4 --num_devices 4 \
                --dataset_name lpm-24-extra --model_config src/configs/config_lpm24_train.yaml --output_folder checkpoints/biot5p_base/SFTBioT5plusBaseContrastive --cuda
```


#### CheBI-20 dataset:
```zsh
python train.py --epochs 50 --batch_size 8 \
                --grad_accum 32 --warmup_ratio 0.05 --lr 1e-4 --num_devices 4 \
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
                      --ckpt_path path/to/ckpt \
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
