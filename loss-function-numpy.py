import tensorflow as tf
import keras
from keras import backend as K
import torch
import torch.nn as nn
import numpy as np

def sigmod(x):
    return 1.0/(1+np.exp(-x))

# logits=np.array([[1.,-810.,20.],[11.,12.,14.],[12.,21.,23.]])
# labels=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
logits = np.random.randn(6, 15, 1)
labels = np.random.randint(2, size=(6, 15, 1)).astype(np.double)

y_predict=sigmod(logits)

loss_1=logits*(1-labels)+np.log(1+np.exp(-logits))
print('公式写的函数\n',loss_1)
print('------------------')
print('tensorflow中的函数\n')
print(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
print('------------------')
print('torch.nn.BCLossWithLogits\n')
print(nn.BCEWithLogitsLoss(reduce=False)(torch.from_numpy(logits), torch.from_numpy(labels)))

################## 总结 ######################################################
'''
pytorch BCEWithLogitsLoss == tensorflow tf.nn.sigmoid_cross_entropy_with_logits 
For brevity, let `x = logits`, `z = labels`.  The logistic loss is
    
          z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + log(1 + exp(-x))
        = x - x * z + log(1 + exp(-x))
'''
