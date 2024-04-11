import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    '''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
    def __init__(self, kl_loss_factor=1.0, T=4.0):
        super(KDLoss, self).__init__()
        self.T = T
        self.kl_loss_factor = kl_loss_factor

    def forward(self, s_out, t_out):
        kd_loss = F.kl_div(F.log_softmax(s_out / self.T, dim=1), 
                           F.softmax(t_out / self.T, dim=1), 
                           reduction='batchmean',
                           ) * self.T * self.T
        return kd_loss * self.kl_loss_factor


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count