import numpy as np
import utils
import evaluate
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Embedding, Conv2D, MaxPool2D, AveragePooling2D, LSTMCell, RNN, BatchNormalization
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import MSE, binary_crossentropy
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras import backend as K
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
from scipy import stats#

class PreTraining:
    def __init__(self, graph, args):
        self.args = args
        self.edges_path = args.edges_list
        self.hidden_size = args.hidden_size
        self.walk_embedding = args.walk_embedding

        self.graph = graph
        graph.load_edgelist(self.edges_path)
        self.adjacency = self.graph.make_adj()
        self.structure = None
        self.emb_dim = args.emb_dim
        self.emb_size = len(graph.nodes)

    def walk_training(self, repeat=100, walk_length=40, trained=False, types=1):
        """
        :param repeat: 每个顶点重复游走的次数
        :param walk_length: 顶点每次游走的长度
        :param trained: trained=false时，表示模型需要重新训练
        :param types: 使用deepwalk时type=1，使用node2vec时type=2
        :return: 返回训练结果embedding
        """
        if trained:
            return np.loadtxt(self.walk_embedding)
        corpus = None
        if types == 1:
            corpus = utils.random_walk(self.adjacency, repeat, walk_length)
        model = Word2Vec(corpus, size=self.emb_dim, window=10, min_count=0, workers=6)

        embedding = np.zeros((self.emb_size, self.emb_dim))
        for i in range(self.emb_size):
            embedding[i] = model[str(i)]
        np.savetxt(self.walk_embedding, embedding)

    def structure_training(self, trained=True):
        pass

class AutoEncoder(object):
    def __init__(self, args, hidden_size, data, path):
        self.data = data
        self.embedding_path = path

        self.data_size = data.shape[0]
        self.data_dim = data.shape[1]
        self.hidden_size = hidden_size
        self.emb_dim = hidden_size[-1]

        self.epoch = args.epoch
        self.folds = args.folds
        self.batch_size = args.batch_size

    def create_model(self):
        X = Input(shape=(self.data_dim,))

        hidden = X
        for size in self.hidden_size[:-1]:
            hidden = Dense(size, activation='relu')(hidden)
        hidden = Dense(self.hidden_size[-1], activation='relu', name='emb')(hidden)
        Y = hidden
        # decode
        for size in reversed(self.hidden_size[:-1]):
            hidden = Dense(size, activation='relu')(hidden)
        X_ = Dense(self.data_dim, activation='sigmoid')(hidden)

        model = Model(inputs=X, outputs=X_)
        encode = Model(inputs=X, outputs=Y)
        return model, encode

    def train(self):
        test_pred = np.zeros_like(self.data)
        embedding = np.zeros((self.data_size, self.emb_dim))
        # train
        kf = KFold(n_splits=self.folds)
        for fold_n, (train_index, test_index) in enumerate(kf.split(self.data)):
            print("fold: {}".format(fold_n + 1))
            opt = Adam(0.01)
            self.model, self.emb = self.create_model()
            self.model.compile(optimizer=opt, loss=evaluate.loss_autoencoder)

            patient = 0
            best_score = 0
            x_train, x_test = self.data[train_index], self.data[test_index]
            # epoch
            # if fold_n == 0:
            #     continue
            for epoch in range(self.epoch):
                # batch
                generator = utils.batch_iter(x_train, self.batch_size, 1)
                for index in generator:
                    self.model.train_on_batch(x_train[index], x_train[index])

                # save best reconsitution model and embedding model
                score, best_score, patient = self.save_best_model(best_score, x_test, patient)
                if (patient > 15 and best_score > 0.7) or patient > 30:
                    break
                # if patient > 10:
                #     break
                print("epoch:{}, score:{}".format(epoch + 1, score))

            self.model = load_model('../data/output/model.h5', custom_objects={'loss_autoencoder':evaluate.loss_autoencoder})
            test_pred[test_index] += self.model.predict(x_test)
            embedding += self.embedding()

        embedding /= self.folds
        np.savetxt(self.embedding_path, embedding)

        # evaluate
        evaluate.recontruction(test_pred, self.data)

    def embedding(self):
        emb = load_model('../data/output/emb_model.h5')
        embedding = emb.predict(self.data)

        return embedding

    def save_best_model(self, best_score, data, patient):
        y_true = data.reshape(-1)
        y_pred = self.model.predict(data).reshape(-1)

        #score = evaluate.f1_binary(y_true,y_pred)
        score = roc_auc_score(y_true, y_pred)
        patient += 1
        if score > best_score:
            patient = 0
            best_score = score
            self.model.save('../data/output/model.h5')
            self.emb.save('../data/output/emb_model.h5')

        return score, best_score, patient
