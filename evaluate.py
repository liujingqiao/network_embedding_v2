from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import utils
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf


def f1(y_true, y_pred):
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return micro_f1, macro_f1

def loss_autoencoder(y_true, y_pred):
    # y_true = tf.reshape(y_true, shape=[-1])
    # y_pred = tf.reshape(y_pred, shape=[-1])
    beta, eps = 10, 0.00001
    label_smoothing = 0.01
    y_true = y_true * (beta - label_smoothing) + 0.5 * label_smoothing
    y_pred = y_pred + eps
    loss = -tf.reduce_mean(y_true*tf.log(y_pred))
    loss = binary_crossentropy(y_true,y_pred)
    return loss

def recontruction(data, adj):
    data = np.array(data)
    y_true, y_pred = None, None
    if len(np.shape(data)) > 1:
        y_pred = data.reshape(-1)
        y_true = adj.reshape(-1)
    score = roc_auc_score(y_true, y_pred)
    print("recontruction auc score : {}".format(score))
    return score

def first_order_loss(y_true, y_pred):
    y_true = np.where(y_true==1, 5, -1)
    loss = K.sum(y_true*K.log(y_pred))
    return loss