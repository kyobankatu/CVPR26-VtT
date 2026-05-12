import random
import argparse
import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--shots', default=8, type=int)
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--mamba_lr', default=5e-4, type=float)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='vision')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA')
    parser.add_argument('--r', default=16, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=8, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    parser.add_argument('--dataset', type=str, default='ISIC')
    parser.add_argument('--image_size', default=224, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--beta', type=float, default=7)
    parser.add_argument('--grad_steps', type=int, default=50)
    parser.add_argument('--prompt_template', type=str, default='a photo of a {}.', help='prompt template used to build class text')
    parser.add_argument('--lambda_adv', type=float, default=0.0, help='weight for token-space adversarial loss on image_mae_encode')
    parser.add_argument('--disc_lr', type=float, default=1e-4, help='learning rate for the token discriminator')
    parser.add_argument('--lambda_align', type=float, default=0.01, help='weight for visual-alignment loss on image_mae_encode')
    parser.add_argument('--lambda_var', type=float, default=0.001, help='weight for variance regularization on image_mae_encode')

    args = parser.parse_args()

    return args
