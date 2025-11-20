import numpy as np
import os
import torch
from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from difflib import SequenceMatcher
import nltk
from rdkit import Chem
from tqdm import tqdm

# Nếu có mô-đun Text2MolMLP
try:
    from .text2mol import Text2MolMLP
except ImportError:
    Text2MolMLP = None

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class Mol2Text_translation:
    def __init__(self, device='cpu', text_model='allenai/scibert_scivocab_uncased', eval_text2mol=False):
        self.text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
        self.eval_text2mol = eval_text2mol
        if eval_text2mol and Text2MolMLP is not None:
            self.text2mol_model = Text2MolMLP(
                ninp=768,
                nhid=600,
                nout=300,
                model_name_or_path=text_model,
                cid2smiles_path=os.path.join(os.path.dirname(__file__), 'ckpts', 'cid_to_smiles.pkl'),
                cid2vec_path=os.path.join(os.path.dirname(__file__), 'ckpts', 'test.txt')
            )
            self.device = torch.device(device)
            self.text2mol_model.load_state_dict(
                torch.load(
                    os.path.join(os.path.dirname(__file__), 'ckpts', 'test_outputfinal_weights.320.pt'),
                    map_location=self.device
                ),
                strict=False
            )
            self.text2mol_model.to(self.device)
        
    def __norm_smile_to_isomeric(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def __levenshtein(self, s1, s2):
        """Chuẩn hoá chuỗi và tính độ tương đồng Levenshtein (dạng similarity)."""
        matcher = SequenceMatcher(None, s1, s2)
        return matcher.ratio()  # từ 0 đến 1

    def __call__(self, predictions, references, smiles=None, text_trunc_length=512):
        meteor_scores = []
        text2mol_scores = []
        exact_match_scores = []
        levenshtein_scores = []

        refs_tokenized = []
        preds_tokenized = []

        if self.eval_text2mol:
            zip_iter = zip(references, predictions, smiles)
        else:
            zip_iter = zip(references, predictions)

        for t in tqdm(zip_iter):
            if self.eval_text2mol:
                gt, out, smile = t
            else:
                gt, out = t

            gt_tokens = self.text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                                    padding='max_length')
            gt_tokens = [tok for tok in gt_tokens if tok not in ['[PAD]', '[CLS]', '[SEP]']]
            out_tokens = self.text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                                     padding='max_length')
            out_tokens = [tok for tok in out_tokens if tok not in ['[PAD]', '[CLS]', '[SEP]']]

            refs_tokenized.append([gt_tokens])
            preds_tokenized.append(out_tokens)

            # METEOR
            meteor_scores.append(meteor_score([gt_tokens], out_tokens))

            # Exact Match
            exact_match_scores.append(1.0 if gt.strip() == out.strip() else 0.0)

            # Levenshtein (theo mức độ giống nhau)
            levenshtein_scores.append(self.__levenshtein(gt, out))

            # Text2Mol (tùy chọn)
            if self.eval_text2mol:
                t2m_score = self.text2mol_model(
                    self.__norm_smile_to_isomeric(smile), out, self.device
                ).detach().cpu().item()
                text2mol_scores.append(t2m_score)

        # BLEU metrics
        bleu = corpus_bleu(refs_tokenized, preds_tokenized)  # BLEU tổng quát
        bleu2 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(.5, .5))
        bleu4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(.25, .25, .25, .25))
        meteor_mean = np.mean(meteor_scores)
        exact_match = np.mean(exact_match_scores)
        levenshtein = np.mean(levenshtein_scores)
        text2mol = np.mean(text2mol_scores) if self.eval_text2mol else None

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores = [scorer.score(out, gt) for gt, out in zip(references, predictions)]
        rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
        rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
        rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

        return {
            "bleu": bleu,
            "bleu2": bleu2,
            "bleu4": bleu4,
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
            "meteor": meteor_mean,
            "exact_match": exact_match,
            "levenshtein": levenshtein,
            "text2mol": text2mol,
        }
