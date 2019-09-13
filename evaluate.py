from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import utils
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.ops import math_ops
import numpy as np
import tensorflow as tf


class Evaluate:
    def __init__(self, beta=None, name=None):
        self.name = name
        self.beta = beta

    def loss(self):
        if not self.name:
            return self.loss_high_order

    @staticmethod
    def f1(y_true, y_pred):
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        return micro_f1, macro_f1

    def loss_high_order(self, y_true, y_pred):
        # y_true = tf.reshape(y_true, shape=[-1])
        # y_pred = tf.reshape(y_pred, shape=[-1])
        label_smoothing, eps = 0.01, 0.00001
        y_true = y_true * self.beta + label_smoothing
        y_pred = y_pred + eps
        # loss = -tf.reduce_mean(y_true*tf.log(y_pred))
        loss = binary_crossentropy(y_true, y_pred)
        return loss

    def recontruction(self, data, adj):
        data = np.array(data)
        y_true, y_pred = None, None
        if len(np.shape(data)) > 1:
            y_pred = data.reshape(-1)
            y_true = adj.reshape(-1)
        score = roc_auc_score(y_true, y_pred)
        print("recontruction auc score : {}".format(score))
        return score

    @staticmethod
    def first_order_loss(y_true, y_pred):
        y_true = math_ops.cast(y_true, y_pred.dtype)
        b = tf.where(y_true == 1.0, 3.0, 1)
        loss = K.mean(K.square(y_true - y_pred)*b)
        return loss
