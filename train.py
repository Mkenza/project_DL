import pickle
import os
import time
import shutil

import torch

import reid_data
from model import ReIdModel
from tqdm import tqdm

import logging


import argparse

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/',
                        help='path to datasets')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    opt = parser.parse_args()
    print(opt)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    model = ReIdModel(opt)
    train_loader = reid_data.get_loader(opt.data_path, batch_size=opt.batch_size)
    # switch to train mode
    model.train()
    for epoch in range(opt.num_epochs):
        for i, train_data in enumerate(tqdm(train_loader)):
            model.train()
            
            # Update the model
            model.train_emb(*train_data)
    
            # # Print log info
            # logging.info(
            #         'Epoch: [{0}][{1}/{2}]\t'
                    
            #         .format(
            #             epoch, i, len(train_loader)))
if __name__ == '__main__':
    main()