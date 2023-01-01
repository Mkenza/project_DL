
import reid_data
import torch
from model import ReIdModel
from tqdm import tqdm
import numpy as np
from utils import LogCollector
import logging
import tensorboard_logger as tb_logger

import argparse
def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='Market_data/Market-1501-v15.09.15/',
                        help='path to folder containing train and test datasets (folders named bounding_box_{train/test}/')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Triplet loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch. final batch size will be 3*batch_size')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the final embedding.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The pretrained CNN
                        (e.g. vgg19)""")
    opt = parser.parse_args()
    print(opt)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


    model = ReIdModel(opt)
    model.logger = LogCollector()

    train_loader = reid_data.get_loader(opt.data_path+'bounding_box_train', batch_size=opt.batch_size)
    val_loader = reid_data.get_loader(opt.data_path+'bounding_box_test', batch_size=opt.batch_size)

    # switch to train mode
    model.train()
    
    r1_list = []
    r1_mean_list = []
    loss_train = []
    r1_best = 0
    for epoch in range(opt.num_epochs):
        print(f"Epoch {epoch+1}/{opt.num_epochs}: \n")
        for i, train_data in enumerate(tqdm(train_loader)):
            model.train()
            
            # Update the model
            model.train_emb(*train_data)
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)
        print("training loss: ", model.loss_t.item())
        loss_train.append(model.loss_t.item())
        _, _, r1, r1_mean = encode_data(model, val_loader)
        # print("Recall@1 on all 13184 images: ", r1)
        r1_mean_list.append(r1_mean)
        r1_list.append(r1)
        if r1>r1_best:
            torch.save(model.state_dict(), f"model_{opt.cnn_type}.PTH")

def encode_data(model, data_loader):
    """encoder des images avec un modèle et un dataset donné par le dataloader
    l'encodage se fera par batch
    returns:
    ---------------
    img_embs: images encodées
    ids: liste des identités correspondantes aux images
    r1: recall@1 sur tout le dataset
    r1_mean: recall@1 moyen sur les batches"""
    # switch to evaluate mode
    model.eval()

    print("encoding...")
    # numpy array to keep all the embeddings
    img_embs = None
    r1 = 0
    ids_final = ["" for i in range(len(data_loader.dataset))]
    for i, (images, indexes, ids) in enumerate(tqdm(data_loader)):
        # compute the embeddings
        img_emb = model.forward(images)
        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[indexes] = img_emb.data.cpu().numpy().copy()
        for i, idx in enumerate(indexes):
          ids_final[idx] = ids[i]

        r1 += recall(img_emb.data.cpu().numpy().copy(), ids)
        # # measure accuracy and record loss
        model.forward_loss(img_emb, ids)

        del images
    
    print("mean recall@1 on batches of validation: ", r1/len(data_loader))
    return img_embs, ids_final, recall(img_embs, ids_final), r1/len(data_loader)

def recall(images, ids):
    """
    Calcul du recall@1
    ne tient pas compte de la position de l'image originale dans le ranking (similarité =1 maximale)

    C'est possible d'avoir une image positive classées mieux que l'image originale,
        on veut pas pénaliser le modèle sur cela
    """
    ranks = np.zeros(images.shape[0])
    top1 = np.zeros(images.shape[0])
    scores = np.dot(images, images.T)
    for index in range(images.shape[0]):
        id = ids[index]
        d = scores[index] # shape (1, batch_size) = distance of positive instances with the rest
        inds = np.zeros(d.shape[0]) 
        inds = np.argsort(d)[::-1]
        for i, ind in enumerate(inds):
          if (ids[ind] == id) and (ind!=index): # on ne regarde pas la distance avec l'élément lui-même
            ranks[index] = i
            break

    # Compute metrics

    # rank could be 0, or 1;
    # 0 if positive example is ranked better than original image
    # 1 if it's right after original image
    r1 = 100.0 * len(np.where(ranks <= 1)[0]) / len(ranks)
    return r1

if __name__ == '__main__':
    tb_logger.configure("logger", flush_secs=5)   
    main()