import numpy as np
import math

import torch 
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Any, Tuple
import utils

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)



class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size, mode='FC'):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        assert mode in ['FC', 'resnet18']
        self.mode = mode
        if mode == 'resnet18':
            self.g = utils.get_model('resnet18', pretrain=True)
            #print(f'resnet.out_features : {self.g.out_features}') # 512
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
            self.p_mu = nn.Sequential(nn.Linear(512, y_dim))
            # p_logvar outputs log of variance of q(Y|X)
            self.p_logvar = nn.Sequential(
                                        nn.Linear(512, y_dim),
                                        nn.Tanh())
        elif mode == 'FC':     
            self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size//2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size//2, y_dim))
            # p_logvar outputs log of variance of q(Y|X)
            self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size//2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size//2, y_dim),
                                        nn.Tanh())
        



    def get_mu_logvar(self, x_samples):
        if self.mode == 'FC':
            x_samples = x_samples[:,0].view(x_samples.size(0), -1)
            mu = self.p_mu(x_samples)
            logvar = self.p_logvar(x_samples)
        elif self.mode == 'resnet18':
            # print(x_samples.shape)
            f_x = self.pool_layer(self.g(x_samples)) # 36*512
            mu = self.p_mu(f_x)
            logvar = self.p_logvar(f_x)

        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        # print(f'mu : {mu}  \n logvar : {logvar}')
        # return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0) / y_samples.size(1)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
   



class CLUBMean(nn.Module):  # Set variance of q(y|x) to 1, logvar = 0. Update 11/26/2022
    def __init__(self, x_dim, y_dim, hidden_size=None):
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        
        super(CLUBMean, self).__init__()
   
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))


    def get_mu_logvar(self, x_samples):
        # variance is set to 1, which means logvar=0
        x_samples = x_samples.view(x_samples.size(0), -1)
        mu = self.p_mu(x_samples)
        return mu, 0
    
    def forward(self, x_samples, y_samples):

        mu, _ = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2.
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2.

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2).sum(dim=1).mean(dim=0) / y_samples.size(1)
    
    def learning_loss(self, x_samples, y_samples): 
        return - self.loglikeli(x_samples, y_samples)
    
    
    


    