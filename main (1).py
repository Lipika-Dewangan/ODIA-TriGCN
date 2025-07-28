import argparse
import torch
import random
import numpy as np
from transformers import BertTokenizer, BertModel

from dataset import load_data, create_dataloader
from model import OdiaTriGCN
from train import train, evaluate
from utils import set_seed, load_config

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Odia-TriGCN: ASTE in Odia")
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode: train or eval')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config['seed'])

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and multilingual BERT
    tokenizer = BertTokenizer.from_pretrained(config['bert_model'])
    bert_model = BertModel.from_pretrained(config['bert_model'])

    # Load dataset
    train_data, val_data, test_data = load_data(config, tokenizer)
    train_loader = create_dataloader(train_data, config, shuffle=True)
    val_loader = create_dataloader(val_data, config)
    test_loader = create_dataloader(test_data, config)

    # Initialize model
    model = OdiaTriGCN(config, bert_model).to(device)

    if args.mode == 'train':
        train(model, train_loader, val_loader, config, device)
    elif args.mode == 'eval':
        evaluate(model, test_loader, config, device)

if __name__ == '__main__':
    main()
