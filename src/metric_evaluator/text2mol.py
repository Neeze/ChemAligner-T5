import os
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import nltk

from difflib import SequenceMatcher
from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit import DataStructs, RDLogger

import selfies as sf
from selfies.exceptions import DecoderError
from tqdm import tqdm

try:
    from .text2mol import Text2MolMLP
except ImportError:
    Text2MolMLP = None

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
RDLogger.DisableLog("rdApp.*")


class Text2MolMetrics:
    """Compute text and molecule-level metrics, compatible with Mol2Text_translation."""

    def __init__(
        self,
        device: str = "cpu",
        text_model: str = "allenai/scibert_scivocab_uncased",
        eval_text2mol: bool = False,
        fcd_fn=None,
    ) -> None:
        """Initialize tokenizer, optional Text2Mol model, and FCD function."""
        self.text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
        self.device = torch.device(device)
        self.fcd_fn = fcd_fn

        self.eval_text2mol = eval_text2mol and (Text2MolMLP is not None)
        if self.eval_text2mol:
            self.text2mol_model = Text2MolMLP(
                ninp=768,
                nhid=600,
                nout=300,
                model_name_or_path=text_model,
                cid2smiles_path=os.path.join(os.path.dirname(__file__), "ckpts", "cid_to_smiles.pkl"),
                cid2vec_path=os.path.join(os.path.dirname(__file__), "ckpts", "test.txt"),
            )
            self.text2mol_model.load_state_dict(
                torch.load(
                    os.path.join(os.path.dirname(__file__), "ckpts", "test_outputfinal_weights.320.pt"),
                    map_location=self.device,
                ),
                strict=False,
            )
            self.text2mol_model.to(self.device)
        else:
            self.text2mol_model = None
            self.eval_text2mol = False

    def __norm_smile_to_isomeric(self, smi: str) -> str:
        """Normalize SMILES to isomeric canonical SMILES."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def __canonical_smiles(self, smi: Optional[str]) -> Optional[str]:
        """Convert SMILES to non-isomeric canonical SMILES, or None if invalid."""
        if smi is None:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

    def __selfies_to_smiles(self, selfies_str: str) -> Optional[str]:
        """Decode a SELFIES string to SMILES, returning None on failure."""
        try:
            return sf.decoder(selfies_str)
        except DecoderError:
            return None
        except Exception:
            return None

    def __seq_similarity(self, s1: str, s2: str) -> float:
        """Compute sequence similarity ratio between two strings."""
        matcher = SequenceMatcher(None, s1, s2)
        return matcher.ratio()

    def __compute_fingerprint_sims(
        self,
        smiles_gt: List[Optional[str]],
        smiles_pred: List[Optional[str]],
        morgan_r: int = 2,
    ) -> Dict[str, Optional[float]]:
        """
        Compute MACCS, RDK, Morgan Tanimoto similarities and validity
        (BioT5+ style).

        validity ≈ 1 - (# invalid predicted SMILES) / (# valid (gt, pred) pairs)
        """
        if len(smiles_gt) != len(smiles_pred):
            raise ValueError("smiles_gt and smiles_pred must have the same length.")

        maccs_sims: List[float] = []
        rdk_sims: List[float] = []
        morgan_sims: List[float] = []

        bad_mols = 0  # số prediction SMILES không canonical/không đọc được
        outputs = []  # chứa (m_gt, m_pr) hợp lệ cho việc tính similarity

        for gt_smi, pr_smi in zip(smiles_gt, smiles_pred):
            # canonical giống BioT5: non-isomeric canonical SMILES
            cg = self.__canonical_smiles(gt_smi)
            cp = self.__canonical_smiles(pr_smi)

            # prediction invalid -> tăng bad_mols và bỏ dòng này
            if cp is None:
                bad_mols += 1
                continue

            # ground truth invalid -> bỏ, KHÔNG tăng bad_mols (BioT5 cũng làm vậy)
            if cg is None:
                continue

            m_gt = Chem.MolFromSmiles(cg)
            m_pr = Chem.MolFromSmiles(cp)
            if m_gt is None or m_pr is None:
                # tương tự BioT5: các case này đã bị loại ở bước canonical trong code gốc,
                # ở đây cứ bỏ qua, không cộng thêm vào bad_mols
                continue

            outputs.append((m_gt, m_pr))

        # Không còn cặp nào hợp lệ
        if not outputs:
            return {
                "validity": 0.0,
                "maccs_fts": None,
                "rdk_fts": None,
                "morgan_fts": None,
            }

        # validity kiểu BioT5+: 1 - bad_mols / len(outputs)
        validity = 1.0 - (bad_mols / len(outputs))

        # Tính fingerprint similarities trên các cặp hợp lệ
        for m_gt, m_pr in outputs:
            maccs_gt = MACCSkeys.GenMACCSKeys(m_gt)
            maccs_pr = MACCSkeys.GenMACCSKeys(m_pr)
            maccs_sims.append(
                DataStructs.FingerprintSimilarity(
                    maccs_gt, maccs_pr, metric=DataStructs.TanimotoSimilarity
                )
            )

            rdk_gt = Chem.RDKFingerprint(m_gt)
            rdk_pr = Chem.RDKFingerprint(m_pr)
            rdk_sims.append(
                DataStructs.FingerprintSimilarity(
                    rdk_gt, rdk_pr, metric=DataStructs.TanimotoSimilarity
                )
            )

            fp_gt = AllChem.GetMorganFingerprint(m_gt, morgan_r)
            fp_pr = AllChem.GetMorganFingerprint(m_pr, morgan_r)
            morgan_sims.append(DataStructs.TanimotoSimilarity(fp_gt, fp_pr))

        maccs_mean = float(np.mean(maccs_sims)) if maccs_sims else None
        rdk_mean = float(np.mean(rdk_sims)) if rdk_sims else None
        morgan_mean = float(np.mean(morgan_sims)) if morgan_sims else None

        return {
            "validity": validity,
            "maccs_fts": maccs_mean,
            "rdk_fts": rdk_mean,
            "morgan_fts": morgan_mean,
        }

    def __compute_fcd(
        self,
        smiles_gt: List[str],
        smiles_pred: List[str],
    ) -> Optional[float]:
        """Compute Fréchet ChemNet Distance using the provided fcd_fn."""
        if self.fcd_fn is None:
            return None
        return float(self.fcd_fn(smiles_gt, smiles_pred))

    def __call__(
        self,
        predictions: List[str],
        references: List[str],
        smiles: Optional[List[str]] = None,
        selfies_gt: Optional[List[str]] = None,
        selfies_pred: Optional[List[str]] = None,
        smiles_gt: Optional[List[Optional[str]]] = None,
        smiles_pred: Optional[List[Optional[str]]] = None,
        text_trunc_length: int = 512,
    ) -> Dict[str, Any]:
        """
        Compute text metrics (BLEU/METEOR/ROUGE/...) and molecule metrics (FTS, FCD).

        predictions, references: text sequences (e.g. captions or SELFIES).
        smiles: SMILES list for Text2Mol score (optional, for eval_text2mol=True).
        selfies_gt, selfies_pred: SELFIES lists (optional, for decoding to SMILES).
        smiles_gt, smiles_pred: SMILES lists (optional, for FTS/FCD).
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length.")

        meteor_scores = []
        text2mol_scores = []
        exact_match_scores = []
        levenshtein_scores = []

        refs_tokenized = []
        preds_tokenized = []

        if self.eval_text2mol:
            if smiles is None or len(smiles) != len(predictions):
                raise ValueError("For eval_text2mol=True, 'smiles' must match predictions length.")
            zip_iter = zip(references, predictions, smiles)
        else:
            zip_iter = zip(references, predictions)

        for t in tqdm(zip_iter):
            if self.eval_text2mol:
                gt, out, smi = t
            else:
                gt, out = t
                smi = None

            gt_tokens = self.text_tokenizer.tokenize(
                gt,
                truncation=True,
                max_length=text_trunc_length,
                padding="max_length",
            )
            gt_tokens = [tok for tok in gt_tokens if tok not in ["[PAD]", "[CLS]", "[SEP]"]]

            out_tokens = self.text_tokenizer.tokenize(
                out,
                truncation=True,
                max_length=text_trunc_length,
                padding="max_length",
            )
            out_tokens = [tok for tok in out_tokens if tok not in ["[PAD]", "[CLS]", "[SEP]"]]

            refs_tokenized.append([gt_tokens])
            preds_tokenized.append(out_tokens)

            meteor_scores.append(meteor_score([gt_tokens], out_tokens))
            exact_match_scores.append(1.0 if gt.strip() == out.strip() else 0.0)
            levenshtein_scores.append(self.__seq_similarity(gt, out))

            if self.eval_text2mol and self.text2mol_model is not None and smi is not None:
                norm_smi = self.__norm_smile_to_isomeric(smi)
                t2m_score = self.text2mol_model(norm_smi, out, self.device).detach().cpu().item()
                text2mol_scores.append(t2m_score)

        bleu = corpus_bleu(refs_tokenized, preds_tokenized)
        bleu2 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.5, 0.5))
        bleu4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.25, 0.25, 0.25, 0.25))

        meteor_mean = float(np.mean(meteor_scores)) if meteor_scores else 0.0
        exact_match = float(np.mean(exact_match_scores)) if exact_match_scores else 0.0
        levenshtein = float(np.mean(levenshtein_scores)) if levenshtein_scores else 0.0

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        rouge_scores = [scorer.score(out, gt) for gt, out in zip(references, predictions)]
        rouge_1 = float(np.mean([rs["rouge1"].fmeasure for rs in rouge_scores])) if rouge_scores else 0.0
        rouge_2 = float(np.mean([rs["rouge2"].fmeasure for rs in rouge_scores])) if rouge_scores else 0.0
        rouge_l = float(np.mean([rs["rougeL"].fmeasure for rs in rouge_scores])) if rouge_scores else 0.0

        text2mol = float(np.mean(text2mol_scores)) if text2mol_scores else None

        if (smiles_gt is None or smiles_pred is None) and (selfies_gt is not None and selfies_pred is not None):
            if len(selfies_gt) != len(selfies_pred):
                raise ValueError("selfies_gt and selfies_pred must have the same length.")
            smiles_gt = [self.__selfies_to_smiles(s) for s in selfies_gt]
            smiles_pred = [self.__selfies_to_smiles(s) for s in selfies_pred]

        if smiles_gt is None or smiles_pred is None:
            maccs_fts = rdk_fts = morgan_fts = validity = None
            fcd = None
        else:
            fp_res = self.__compute_fingerprint_sims(smiles_gt, smiles_pred)
            validity = fp_res["validity"]
            maccs_fts = fp_res["maccs_fts"]
            rdk_fts = fp_res["rdk_fts"]
            morgan_fts = fp_res["morgan_fts"]

            valid_smiles_gt = []
            valid_smiles_pred = []
            for gt_smi, pr_smi in zip(smiles_gt, smiles_pred):
                cg = self.__canonical_smiles(gt_smi)
                cp = self.__canonical_smiles(pr_smi)
                if cg is None or cp is None:
                    continue
                if Chem.MolFromSmiles(cg) is None or Chem.MolFromSmiles(cp) is None:
                    continue
                valid_smiles_gt.append(cg)
                valid_smiles_pred.append(cp)
            if valid_smiles_gt and valid_smiles_pred:
                fcd = self.__compute_fcd(valid_smiles_gt, valid_smiles_pred)
            else:
                fcd = None

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
            "validity": validity,
            "maccs_fts": maccs_fts,
            "rdk_fts": rdk_fts,
            "morgan_fts": morgan_fts,
            "fcd": fcd,
        }
