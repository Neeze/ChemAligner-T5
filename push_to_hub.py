import torch
import transformers
from huggingface_hub import login
from transformers import AutoTokenizer
from transformers.models.t5 import T5ForConditionalGeneration
from argparse import ArgumentParser, Namespace
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
device = 'cuda'

def main(args):
    login(args.hf_token)
    tokenizer = AutoTokenizer.from_pretrained('QizhiPei/biot5-plus-base')
    model = T5ForConditionalGeneration.from_pretrained(
        'QizhiPei/biot5-plus-base'
    )
    name_or_path_hub = f'Neeze/{args.model_name}'
    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location="cpu")
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    print(f"Original state_dict keys: {len(state_dict)}")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("t5_model."):
            new_state_dict[k[len("t5_model."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    tokenizer.push_to_hub(name_or_path_hub, private=True)
    model.push_to_hub(name_or_path_hub, private=True, safe_serialization=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='biot5-plus-base-sft')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    args = parser.parse_args()
    main(args)