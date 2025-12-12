import torch
from transformers import AutoTokenizer
from src.dataset_module import get_dataloaders
import lightning as pl
import optuna  # <---- thêm

from src.lighning_module_base import T5Model as T5ModelBase
from src.lightning_module import T5Model as T5ModelContrastive
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser, Namespace
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
import yaml
import os
import math
from src.utils import set_nested_attr

METHOD_MAP = {
    'base': T5ModelBase,
    'chemaligner': T5ModelContrastive,
}


def train_once(args, trial=None):
    # cố định seed cho mỗi lần train (có thể thêm trial.number nếu muốn random theo trial)
    seed_everything(2183)

    device = torch.device('cuda' if args.cuda else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)

    if args.dataset_name == 'lpm-24':
        args.dataset_name_or_path = 'duongttr/LPM-24-extend'
    elif args.dataset_name == 'lpm-24-extra':
        args.dataset_name_or_path = 'Neeze/LPM-24-extra-extend'
    elif args.dataset_name == 'lpm-24-smoke':
        args.dataset_name_or_path = 'Neeze/LPM-24-smoke-extend'
    elif args.dataset_name == 'chebi-20':
        args.dataset_name_or_path = 'duongttr/chebi-20-new'
    else:
        raise Exception('Dataset name is invalid, please choose one in two: lpm-24, chebi-20')

    train_dataloader = get_dataloaders(
        args, tokenizer, batch_size=args.batch_size,
        num_workers=args.num_workers, split='train', task='lang2mol'
    )
    val_dataloader = get_dataloaders(
        args, tokenizer, batch_size=args.batch_size,
        num_workers=args.num_workers, split='validation', task='lang2mol'
    )

    args.train_data_len = len(train_dataloader) // args.grad_accum
    args.tokenizer = Namespace()
    args.tokenizer.pad_token_id = tokenizer.pad_token_id

    T5Model = METHOD_MAP[args.method]
    model = T5Model(args)
    model.to(device)
    model.tokenizer = tokenizer

    on_best_eval_loss_callback = ModelCheckpoint(
        dirpath=args.output_folder,
        filename='ckpt_{epoch}_{eval_loss}',
        save_top_k=3,
        verbose=True,
        monitor='eval_loss',
        mode='min'
    )

    wandb_logger = WandbLogger(
        log_model=False,
        project='ACL_Lang2Mol',
        name=os.path.splitext(os.path.basename(args.model_config))[0] + f"_{args.method}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [on_best_eval_loss_callback, lr_monitor]

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=args.epochs,
        accelerator='cuda' if args.cuda else 'cpu',
        strategy='ddp_find_unused_parameters_true' if args.num_devices > 1 else 'auto',
        devices=args.num_devices,
        precision=args.precision,  # 32 if has more vram
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=[wandb_logger],
        accumulate_grad_batches=args.grad_accum,
        deterministic=True
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    if trial is not None:
        eval_loss = trainer.callback_metrics["eval_loss"].item()
        return eval_loss


def main(args):
    if args.optuna:
        def objective(trial):
            trial_args = Namespace(**vars(args))
            trial_args.lr = trial.suggest_categorical(
                "lr", [5e-5, 1e-4, 2e-4, 5e-4]
            )

            print(f"[Trial {trial.number}] lr = {trial_args.lr}")
            return train_once(trial_args, trial)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials)

        print("Best trial:")
        best_trial = study.best_trial
        print(f"  lm_loss = {best_trial.value}")
        print("  params:")
        for k, v in best_trial.params.items():
            print(f"    {k}: {v}")
    else:
        train_once(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default='base', choices=['base', 'chemaligner'],
                        help="Select method type.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--dataset_name', type=str, default='lpm-24')
    parser.add_argument('--model_config', type=str, default='src/configs/config_lpm24_train.yaml')
    parser.add_argument('--output_folder', type=str, default='weights/')
    parser.add_argument('--optuna', action='store_true', help="Use Optuna to search best lr.")
    parser.add_argument('--n_trials', type=int, default=10, help="Number of Optuna trials.")

    args = parser.parse_args()
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    for key, value in model_config.items():
        set_nested_attr(args, key, value)

    main(args)