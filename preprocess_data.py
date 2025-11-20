import os
import argparse
from typing import List, Optional

from datasets import load_dataset, DatasetDict, Features, Value, Image
from datasets.utils.py_utils import convert_file_size_to_int
from datasets import config as ds_config

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

from tqdm import tqdm


def init_fail_log(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("split\trow_index\tstage\tsmiles\tmessage\n")


def log_fail(
    log_path: str,
    split: str,
    row_index: int,
    stage: str,
    smiles: str,
    message: str,
):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{split}\t{row_index}\t{stage}\t{smiles}\t{message.replace(os.linesep, ' ')}\n"
        )


def smiles_to_selfies(
    smiles: str, split: str, row_index: int, log_path: str
) -> Optional[str]:
    try:
        selfies_str = sf.encoder(smiles)
        if selfies_str is None:
            log_fail(
                log_path,
                split,
                row_index,
                stage="selfies_encoder_none",
                smiles=smiles,
                message="selfies.encoder returned None",
            )
        return selfies_str
    except Exception as e:
        log_fail(
            log_path,
            split,
            row_index,
            stage="selfies_exception",
            smiles=smiles,
            message=repr(e),
        )
        return None


def smiles_to_image(
    smiles: str, split: str, row_index: int, log_path: str
):
    """
    Dùng RDKit để convert SMILES thành PIL Image size 512x512.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)  # ✅ đúng hàm RDKit
        if mol is None:
            log_fail(
                log_path,
                split,
                row_index,
                stage="rdkit_mol_none",
                smiles=smiles,
                message="Chem.MolFromSmiles returned None",
            )
            return None
        img = Draw.MolToImage(mol, size=(512, 512))
        return img
    except Exception as e:
        log_fail(
            log_path,
            split,
            row_index,
            stage="rdkit_exception",
            smiles=smiles,
            message=repr(e),
        )
        return None


def preprocess_batch(
    batch,
    indices: List[int],
    split_name: str,
    log_path: str,
):
    smiles_list: List[str] = batch["molecule"]
    captions: List[str] = batch["caption"]

    images = []
    smiles_out = []
    captions_out = []
    selfies_out = []

    for row_idx, smiles, cap in zip(indices, smiles_list, captions):
        smiles_out.append(smiles)
        captions_out.append(cap)

        selfies_str = smiles_to_selfies(
            smiles=smiles,
            split=split_name,
            row_index=row_idx,
            log_path=log_path,
        )
        selfies_out.append(selfies_str)

        img = smiles_to_image(
            smiles=smiles,
            split=split_name,
            row_index=row_idx,
            log_path=log_path,
        )
        images.append(img)

    return {
        "image": images,
        "smiles": smiles_out,
        "caption": captions_out,
        "selfies": selfies_out,
    }


def save_split_as_parquet_shards(
    ds,
    split_name: str,
    output_dir: str,
    max_shard_size: str = "435MB",
):
    """
    Lưu 1 split thành nhiều file .parquet trong thư mục output_dir
    sao cho mỗi file <= max_shard_size.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_nbytes = ds._estimate_nbytes()
    max_shard_bytes = convert_file_size_to_int(
        max_shard_size or ds_config.MAX_SHARD_SIZE
    )

    num_shards = int(dataset_nbytes / max_shard_bytes) + 1
    num_shards = max(num_shards, 1)

    print(
        f"[{split_name}] estimated size = {dataset_nbytes/1e9:.2f} GB, "
        f"splitting into {num_shards} shards (max {max_shard_size} each)."
    )

    for shard_index in tqdm(
        range(num_shards), desc=f"Saving {split_name} to parquet shards"
    ):
        shard = ds.shard(
            num_shards=num_shards, index=shard_index, contiguous=True
        )
        shard_path = os.path.join(
            output_dir,
            f"{split_name}-{shard_index:05d}-of-{num_shards:05d}.parquet",
        )
        shard.to_parquet(shard_path)


def push_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    token: Optional[str],
    max_shard_size: str = "435MB",
    private: bool = False,
):
    print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        max_shard_size=max_shard_size,
    )
    print("Done pushing to Hub.")


# ---------- FILTER FUNCS ----------

def is_valid_example(example):
    """
    Lọc:
    - caption / smiles / selfies không được None hoặc rỗng
    - image không được None
    => loại hết các case failed trong quá trình preprocess
    """
    if example["caption"] is None or example["caption"] == "":
        return False
    if example["smiles"] is None or example["smiles"] == "":
        return False
    if example["selfies"] is None or example["selfies"] == "":
        return False
    if example["image"] is None:
        return False
    return True


def make_dedup_filter():
    """
    Lọc duplicate theo (smiles, caption).
    Dùng closure 'seen' nên KHÔNG dùng num_proc > 1 cho bước filter này.
    """
    seen = set()

    def _filter(example):
        key = (example["smiles"], example["caption"])
        if key in seen:
            return False
        seen.add(key)
        return True

    return _filter


# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lpm24_selfies_out",
        help="Thư mục lưu dataset và các file parquet.",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="Repo id trên HF Hub, ví dụ: yourname/LPM-24_train-extra-selfies",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HF token (nếu không truyền thì dùng env HF_TOKEN).",
    )
    parser.add_argument(
        "--max_parquet_size",
        type=str,
        default="435MB",
        help="Giới hạn kích thước mỗi file .parquet, vd '435MB'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Nếu set thì repo trên Hub sẽ là private.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Số process song song để map (dùng nhiều CPU core).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fail_log_path = os.path.join(args.output_dir, "failed_selfies.log")
    init_fail_log(fail_log_path)

    print("Loading original dataset language-plus-molecules/LPM-24_train-extra ...")
    raw = load_dataset("language-plus-molecules/LPM-24_train-extra")

    available_splits = list(raw.keys())
    print(f"Available splits: {available_splits}")

    split_map = {
        "split_train": "train",
        "split_valid": "validation",
    }

    processed_splits = {}

    for raw_split, new_name in split_map.items():
        if raw_split not in raw:
            raise ValueError(
                f"Split '{raw_split}' không tồn tại trong dataset gốc. "
                f"Splits có: {available_splits}"
            )

        ds = raw[raw_split]
        print(
            f"\nPreprocessing split '{raw_split}' ({len(ds)} rows) "
            f"→ '{new_name}' ..."
        )

        processed = ds.map(
            lambda batch, idx: preprocess_batch(
                batch, idx, split_name=new_name, log_path=fail_log_path
            ),
            batched=True,
            with_indices=True,
            remove_columns=ds.column_names,
            desc=f"smiles→selfies + RDKit image ({new_name})",
            num_proc=args.num_proc,
        )

        # Features chuẩn
        features = Features(
            {
                "image": Image(),
                "smiles": Value("string"),
                "caption": Value("string"),
                "selfies": Value("string"),
            }
        )
        processed = processed.cast(features)

        # --- Bước 1: lọc samples invalid ---
        before = len(processed)
        processed = processed.filter(is_valid_example, num_proc=args.num_proc)
        after = len(processed)
        print(f"[{new_name}] valid-filter: {before} -> {after}")

        # --- Bước 2: deduplicate (smiles, caption) ---
        before_dedup = len(processed)
        dedup_fn = make_dedup_filter()
        processed = processed.filter(dedup_fn)  # num_proc=1 mặc định
        after_dedup = len(processed)
        print(f"[{new_name}] dedup: {before_dedup} -> {after_dedup}")

        processed_splits[new_name] = processed

    processed_ds = DatasetDict(processed_splits)

    arrow_dir = os.path.join(args.output_dir, "arrow")
    print(f"\nSaving full DatasetDict (arrow) to {arrow_dir}")
    processed_ds.save_to_disk(
        arrow_dir, max_shard_size=args.max_parquet_size
    )

    # tất cả .parquet nằm trong folder con /data
    parquet_dir = os.path.join(args.output_dir, "data")
    for split_name in processed_ds.keys():
        split_parquet_dir = os.path.join(parquet_dir, split_name)
        save_split_as_parquet_shards(
            processed_ds[split_name],
            split_name=split_name,
            output_dir=split_parquet_dir,
            max_shard_size=args.max_parquet_size,
        )

    print("\n=== DONE PREPROCESSING ===")
    print(f"Arrow dataset dir : {arrow_dir}")
    print(f"Parquet shards dir (under /data): {parquet_dir}")
    print(f"Fail log          : {fail_log_path}")

    token = args.hf_token or os.getenv("HF_TOKEN")
    if token is None:
        print(
            "\n⚠️  Không tìm thấy token. "
            "Set HF_TOKEN trong env hoặc truyền --hf_token để push lên Hub."
        )
    else:
        push_to_hub(
            processed_ds,
            repo_id=args.hf_repo_id,
            token=token,
            max_shard_size=args.max_parquet_size,
            private=args.private,
        )

    print("\nHoàn thành. Bạn có thể xem lại:")
    print(f"- Dataset arrow: {arrow_dir}")
    print(f"- Parquet (<= {args.max_parquet_size} mỗi file) trong folder /data: {parquet_dir}")
    print(f"- Log selfies fail: {fail_log_path}")
    print(
        "- Dataset trên Hub (sau khi push): "
        f"https://huggingface.co/datasets/{args.hf_repo_id}"
    )


if __name__ == "__main__":
    main()