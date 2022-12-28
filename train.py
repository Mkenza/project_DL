
import reid_data
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
    model.logger = LogCollector()

    train_loader = reid_data.get_loader(opt.data_path+'bounding_box_train', batch_size=opt.batch_size)
    val_loader = reid_data.get_loader(opt.data_path+'bounding_box_test', batch_size=opt.batch_size)

    # switch to train mode
    model.train()
    for epoch in range(opt.num_epochs):
        for i, train_data in enumerate(tqdm(train_loader)):
            model.train()
            
            # Update the model
            model.train_emb(*train_data)
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)
        print("training loss: ", model.loss_t.item())
        encode_data(model, val_loader)
def encode_data(model, data_loader):
    # switch to evaluate mode
    model.eval()

    print("encoding...")
    # numpy array to keep all the embeddings
    img_embs = None
    r1 = 0
    for i, (images, indexes, ids) in enumerate(tqdm(data_loader)):

        # compute the embeddings
        img_emb = model.forward(images)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            ids_final = np.zeros(len(data_loader.dataset))
        # preserve the embeddings by copying from gpu and converting to numpy
        ids_final[indexes] = ids
        img_embs[indexes] = img_emb.data.cpu().numpy().copy()
        r1 += recall(img_emb.data.cpu().numpy().copy(), ids)
        # # measure accuracy and record loss
        # loss_val += model.forward_loss(img_emb, ids)

        del images
    
    print("mean recall@1 on batches of validation: ", r1/len(data_loader))
    return img_embs, ids_final

def recall(images, ids):
    """
    Calcul du recall@1
    """
    ranks = np.zeros(images.shape[0])
    top1 = np.zeros(images.shape[0])
    scores = np.dot(images, images.T)
    for index in range(images.shape[0]):
        id = ids[index]
        d = scores[index] # shape (1, batch_size) = distance of positive instances with the rest
        inds = np.zeros(d.shape[0]-1) # on ne garde pas la distance avec l'élément lui-même
        inds = np.argsort(d)[::-1]
        # print(inds)
        for i, ind in enumerate(inds[1:]):
          if ids[ind] == id:
            ranks[index] = i
            break

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    return r1

if __name__ == '__main__':
    tb_logger.configure("logger", flush_secs=5)   
    main()