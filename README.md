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

# install dependency if needed
pip install sconf optuna fcd
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

## 3. Training model (configure everything via YAML)

ChemAligner-T5 is fully **config-driven**.
All training and evaluation behaviors are controlled through `.yaml` files, so users do **not** need to modify Python code to run experiments.

You only need to prepare:

* one **training config** (`*_train.yaml`)
* one **evaluation config** (`*_eval.yaml`)

Both LPM-24 and CheBI-20 use the **same config structure**, differing only in dataset names and paths.

---

### Config train file

The training config defines **model backbone**, **training strategy**, **dataset loading**, **loss design**, and **evaluation behavior during training**.

Below is a structured explanation of each section.

---

#### `t5`

```yaml
t5:
  pretrained_model_name_or_path: "QizhiPei/biot5-plus-base"
```

* Specifies the **base language model**.
* ChemAligner-T5 is built on top of **BioT5+**.
* Any compatible T5-style checkpoint can be substituted here.

---

#### `trainer_init`

Controls **optimization, distributed training, checkpointing, and logging**.

Key fields:

```yaml
trainer_init:
  seed_everything: 2183
  cuda: true
  deterministic: true
  strategy: "ddp_find_unused_parameters_true"
```

* Ensures reproducibility.
* Uses PyTorch Lightning with **DDP** for multi-GPU training.

Checkpointing:

```yaml
  output_folder: "output/chebi20_train"
  filename: "ckpt_{epoch}_{eval_loss}"
  save_top_k: 3
  monitor: "avg_metrics"
  mode: "max"
```

* Automatically saves top-k checkpoints based on the monitored metric.
* When `run_text2mol_metrics = true`, `avg_metrics` is recommended.
* Otherwise, switch to `eval_loss` and `mode: min`.

Training scale:

```yaml
  max_epochs: 100
  num_devices: 4
  grad_accum: 8
  precision: "32"
```

* `grad_accum` enables large effective batch sizes without exceeding GPU memory.
* Precision can be changed to `"16-mixed"` or `"bf16-mixed"` if desired.

Optimizer and scheduler:

```yaml
  optimizer:
    name: "adamw"
    weight_decay: 1e-4
  lr: 5e-5
  warmup_ratio: 0.0
```

Optional hyperparameter search:

```yaml
  optuna:
    activate: false
```

Experiment tracking:

```yaml
  wandb:
    project: "ACL_Lang2Mol"
    name: "chebi20_train"
```

---

#### `dataset_init`

Defines **which dataset to use and how it is loaded**.

```yaml
dataset_init:
  dataset_name: "chebi-20"
  task: "lang2mol"
  train_batch_size: 8
  eval_batch_size: 64
  num_workers: 24
```

* Supported datasets:

  * `lpm-24`
  * `lpm-24-extra`
  * `chebi-20`
* The dataset name is internally mapped to HuggingFace datasets.
* `num_workers` is automatically divided by the number of GPUs.

---

#### `method_init`

Controls **training objective design**.

```yaml
method_init:
  method: "chemaligner"
  seq2seq_loss_weight: 1.0
  contrastive_loss_weight: 0.3
```

* `method: base`
  → standard BioT5+ sequence-to-sequence training
* `method: chemaligner`
  → joint **generation + contrastive representation alignment**

Loss weights allow balancing:

* token-level generation quality
* sequence-level text–molecule alignment

---

#### `eval_init` (during training)

Controls **on-the-fly validation**.

```yaml
eval_init:
  max_length: 512
  num_beams: 1
  use_amp: true
  run_text2mol_metrics: true
```

* Uses greedy decoding by default (`num_beams = 1`).
* `run_text2mol_metrics` enables molecule-level metrics (FCD, similarity, etc.).
* Validation is automatically run at the end of each epoch.

Optional:

```yaml
  run_evaluate_after_trainning_done: false
```

* If enabled, all saved checkpoints will be evaluated automatically after training.

---

### Config validation file

The evaluation config is **checkpoint-centric** and supports **single-GPU or multi-GPU evaluation**, as well as **multiple checkpoints in one run**.

---

#### `t5`

Same as training:

```yaml
t5:
  pretrained_model_name_or_path: "QizhiPei/biot5-plus-base"
```

---

#### `dataset_init`

```yaml
dataset_init:
  dataset_name: "chebi-20"
  split: validation
  eval_batch_size: 240
```

* Uses the validation split by default.
* Batch size is **per process** (per GPU if using DDP).

---

#### `eval_init`

Controls **generation, metrics, and output format**.

```yaml
eval_init:
  num_devices: 1
  cuda: true
  max_length: 512
  num_beams: 1
  use_amp: true
```

Multi-checkpoint evaluation:

```yaml
  checkpoint_paths:
    - ".../ckpt_epoch=10.ckpt"
    - ".../ckpt_epoch=20.ckpt"
    - ".../ckpt_epoch=30.ckpt"
```

* Each checkpoint is evaluated independently.
* Predictions and metrics are saved per checkpoint.
* A unified `metrics_summary.csv` is produced automatically.

Output control:

```yaml
  output_dir: "results/chebi20_evaluation"
  summary_file: "metrics_summary.csv"
```

Optional debug flags:

```yaml
  print_examples: 0
  offline: false
```

---

### Summary

* **Training and evaluation are fully YAML-controlled**
* Supports:

  * single or multi-GPU
  * multiple checkpoints per evaluation
  * sequence-level and molecule-level metrics
* Designed for **reproducible chemical language research**

Once configs are ready, running experiments is as simple as:

```bash
python train.py --model_config src/configs/chebi20_train.yaml
python eval_lang2mol.py --model_config src/configs/chebi20_eval.yaml
```



### LPM-24 dataset:

SFT BioT5+ scripts

```zsh
python train.py --model_config src/configs/lpm24_train.yaml
```

#### Evaluate on LPM-24 dataset
```zsh
python eval_lang2mol.py --model_config src/configs/lpm24_eval.yaml
```

```zsh
torchrun --nproc_per_node=<your devices> --model_config src/configs/lpm24_eval.yaml
```

### CheBI-20 dataset:
```zsh
python train.py --model_config src/configs/chebi20_train.yaml 
```

#### Evaluate on CheBI-20
```zsh
python eval_lang2mol.py --model_config src/configs/chebi20_eval.yaml
```

```zsh
torchrun --nproc_per_node=<your devices> eval_lang2mol.py --model_config src/configs/chebi20_eval.yaml
```

### Push to hub

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
