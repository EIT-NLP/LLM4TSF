import os
import argparse
import datasets
from accelerate import Accelerator
import torch
from model_infer import model_infer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser(description='tsmodels')
parser.add_argument('--seq_len', type=int, default=336)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--d_model', type=int, default=0)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--family', type=str, default="LLM4TS") # LLM4TS or TSFMs
parser.add_argument('--scale', type=str, default='Tiny') # Tiny Small Base Large
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--is_load_checkpoint', type=bool, default=True)
parser.add_argument('--is_save_checkpoint', type=bool, default=False)
parser.add_argument('--fix_seed', type=int, default=2025)
parser.add_argument('--test_metrics_path', type=str, default='main/single_dataset_learning/infer')
parser.add_argument('--outdir_base', type=str,default='main/single_dataset_learning/infer')
args = parser.parse_args()
def run():
    name_list = ["ETTh1","ETTh2","ETTm1","ETTm2","electricity","exchange_rate","Solar","weather"]
    for name in name_list:
        # All training sets from individual datasets are mixed together; similarly, the test sets and validation sets are each combined respectively.
        tr_path = "arrow_train_single/"+name
        va_path = "arrow_va_single/"+name
        te_path = "arrow_test_single/"+name
        dataset_tr = datasets.load_from_disk(tr_path)
        dataset_va = datasets.load_from_disk(va_path)
        dataset_te = datasets.load_from_disk(te_path)
        accelerator = Accelerator()
        if accelerator.is_main_process:
            with open(args.test_metrics_path, "a") as f:
                f.write(f"{name}\n")
        model_infer(dataset_tr,dataset_va,dataset_te,args.test_metrics_path,args)
        torch.cuda.empty_cache()
