import random
import numpy as np
import torch


def backbone_selection(args):
    if args.scale == "Tiny" or args.scale == "Small":
        args.d_model = 768
        args.model_name = "GPT2"
    elif args.scale == "Base":
        args.d_model = 1024
        args.model_name = "Qwen3-0.6B"
    elif args.scale == "Large":
        args.d_model = 2048
        args.model_name = "Qwen3-1.7B"
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)