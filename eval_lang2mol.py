from argparse import ArgumentParser, Namespace
from src.utils import set_nested_attr
import yaml
from src.lightning_module_lang2mol_base import T5Model
from src.dataset_module import get_dataloaders
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import csv
import os
from src.metric_evaluator.translation_metrics import Mol2Text_translation

def main(args):
    evaluator = Mol2Text_translation()
    
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)
    
    if args.dataset_name == 'lpm-24':
        args.dataset_name_or_path = 'duongttr/LPM-24-extend'
    elif args.dataset_name == 'lpm-24-extra':
        args.dataset_name_or_path = 'Neeze/LPM-24-extra-extend'
    elif args.dataset_name == 'lpm-24-eval':
        args.dataset_name_or_path = 'Neeze/LPM-24-eval-extend'
    elif args.dataset_name == 'chebi-20':
        args.dataset_name_or_path = 'duongttr/chebi-20-new'
    else:
        raise Exception('Dataset name is invalid, please choose one in two: lpm-24, chebi-20')
    
    val_dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=4, split=args.split, task='lang2mol')
    
    args.tokenizer = Namespace()
    args.tokenizer.pad_token_id = tokenizer.pad_token_id
    
    model = T5Model(args)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.tokenizer = tokenizer
    
    if args.checkpoint_path:
        model.load_state_dict(
            torch.load(args.checkpoint_path, map_location=device)['state_dict'], strict=False
        )
        
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True) 
    
    ALL_GT = []
    ALL_PRED = []
    with open(args.output_csv, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['caption', 'gt_selfie', 'pred_selfie'])
        writer.writeheader()
        
        for idx, batch in enumerate(tqdm(val_dataloader)):
            for k, v in batch.items():
                try:
                    batch[k] = batch[k].to('cuda')
                except:
                    pass
            # batch = {k:v.to('cuda') if isinstance(v, torch.Tensor) or isinstance(v, ) else v for k,v in batch.items()}
            pred_selfies = model.generate_captioning(batch, decoder_start_token_id=0)
            gt_selfies = batch['selfies']
            
            for p, g in zip(pred_selfies, gt_selfies):
                print(f"Predict: {p}")
                print(f"GT: {g}")
                print('-'*50)
            
            ALL_GT.extend(gt_selfies)
            ALL_PRED.extend(pred_selfies)
            
            writer.writerows([
                {'caption': caption,
                 'gt_selfie': gt_selfie,
                 'pred_selfie': pred_selfie} for caption, gt_selfie, pred_selfie in  zip(batch['caption'], gt_selfies, pred_selfies)
            ])
        
    metrics_result = evaluator(ALL_PRED, ALL_GT)
    for k, v in metrics_result.items():
        print(f'{k}: {v}')

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default='lpm-24')
    parser.add_argument('--model_config', type=str, default='src/configs/config_lpm24_train.yaml')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default='results/output.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--split', type=str, default='validation')
    
    args = parser.parse_args()
    
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    for key, value in model_config.items():
        set_nested_attr(args, key, value)
        
    main(args)