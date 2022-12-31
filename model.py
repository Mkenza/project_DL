from collections import OrderedDict
from scipy import spatial
import torch 
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X
class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)
  
class TripletLoss(nn.Module):
    """
    Triplet loss: exemples positifs = les 5 éléments avec id identique
    exemples négatifs = les éléments d'id différent et parmi les 5 le plus similaires à l'anchor selon le modèle
    """

    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, im, ids):
        # exemples positifs
        positive = np.array([np.where(np.array(ids)==id) for id in np.array(ids)]) # shape (batch_size, vary)
        
        # exemples negatifs
        negative = np.array([np.where(np.array(ids)!=id) for id in np.array(ids)]) # shape (batch_size, vary)
        
        # matrice de similarites entre les images encodees
        scores = im.mm(im.t()) # shape (batch_size, batch_size)
        
        cost = 0

        for j, pos in enumerate(positive):
          # get negatives for j-th anchor
          neg = negative[j]
        
          scores[j][j] = 0
          scores_anchor = scores[j]

          # compare maximum of similarity with negatives, with minimum similarity to positives to loss
          cost += max(self.margin + torch.max(scores_anchor[neg])-torch.min(scores_anchor[pos]), 0)
        return cost

class ReIdModel(nn.Module):

    def __init__(self, opt):
        """Load pretrained VGG16 or RESNET50 and replace top fc layer."""
        super(ReIdModel, self).__init__()
        self.embed_size = opt.embed_size

        # Load a pre-trained model
        self.cnn = self.get_cnn(opt.cnn_type, True)

        
        # Replace the last fully connected layer of CNN with a new one
        if opt.cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                512)
            self.fc2 = nn.Linear(512, opt.embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif opt.cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, opt.embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()
        self.criterion = TripletLoss(margin=opt.margin)
        params = list(self.fc.parameters())
        params += list(self.fc2.parameters())
        params += list(self.cnn.parameters())
        self.params = params
        self.Eiters = 0
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        if torch.cuda.is_available():
          self.cuda()
    def set_logger(self, logger):
        self.logger = logger
    def get_cnn(self, arch, pretrained):
        model = models.__dict__[arch]()
        model.features = nn.DataParallel(model.features)
        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(ReIdModel, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        features = self.fc(features)
        features = self.fc2(F.relu(features))
        features = l2norm(features)
        return features

    def forward_loss(self, img_emb, ids, **kwargs):
        """Compute the triplet loss given image embeddings
        """
        loss = self.criterion(img_emb, ids)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        self.loss_t = loss
        return loss
    
    def train_emb(self, images, indexes, ids, *args):
        """One training step.
        """
        self.Eiters += 1
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Eit', self.Eiters)

        # compute the embeddings
        img_emb= self.forward(images)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, ids)
        # print("training loss: ", loss)
        # compute gradient and do SGD step
        
        loss.backward()
        self.optimizer.step()
