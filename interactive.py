from pre_train import PreTraining
import numpy as np
import utils
import evaluate
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Embedding, Conv2D, MaxPool2D, AveragePooling2D, LSTMCell, RNN, BatchNormalization
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.initializers import RandomNormal
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from evaluate import Evaluate
from scipy import stats
# tf.set_random_seed(10)


class Interactive:
    def __init__(self, graph, args):
        self.args = args
        self.folds = args.folds
        self.classes_path = args.classes_input
        self.edges_path = args.edges_list
        self.batch_size = args.batch_size

        self.graph = graph
        self.emb_dim = args.emb_dim
        self.emb_size = len(graph.nodes)

    def init_embedding(self):
        embedding = dict()
        pre_train = PreTraining(self.graph, self.args)

        embedding['walk'] = pre_train.deep_walk(trained=True)
        embedding['attr'] = pre_train.attributes_proximity(trained=True)
        embedding['stru'] = pre_train.structure_proximity(trained=True)
        embedding['link'] = pre_train.link_proximity(trained=True)
        embedding['walk_stru'] = pre_train.walk_proximity(trained=True)
        embedding['classes'] = pre_train.classes_proximity(trained=True)
        return embedding

    def create_model(self, embedding, types, task='classifier'):
        seed = RandomNormal(mean=0.0, stddev=0.05, seed=42)
        # cora:42
        vi = Input(shape=(), dtype=tf.int32)
        vj = Input(shape=(), dtype=tf.int32)

        # Pre-training output
        walk_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['walk']])
        stru_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['stru']])
        # link_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['link']])
        attr_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['attr']])
        walk_stru_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['walk_stru']])
        classes_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['classes']])

        concat, shape = None, 5
        if task == 'classifier':
            concat = tf.concat([walk_emb(vi), stru_emb(vi), attr_emb(vi), walk_stru_emb(vi)], axis=1)
        if task == 'link':
            concat_vi = tf.concat([walk_emb(vi),stru_emb(vi),attr_emb(vi),classes_emb(vi),walk_stru_emb(vi)], axis=1)
            concat_vj = tf.concat([walk_emb(vj),stru_emb(vj),attr_emb(vj),classes_emb(vi),walk_stru_emb(vj)], axis=1)
            concat = concat_vi*concat_vj
            # concat = tf.concat([walk_emb(vi)], axis=1)

        attention = Dense(concat.shape[1], activation='softmax', kernel_initializer=seed)(concat)
        attention = concat*attention

        reshape = tf.reshape(attention, shape=(-1, shape, self.emb_dim))
        reshape = tf.expand_dims(reshape, -1)
        conv = None
        for i, size in enumerate([[shape, 5], [shape, 3],  [shape, 2]]):
            conv2d = Conv2D(filters=5, kernel_size=size, kernel_initializer=seed, padding='same')(reshape)
            pool = AveragePooling2D(pool_size=(1, 2))(conv2d)
            dim = pool.shape[1]*pool.shape[2]*pool.shape[3]
            conv2d = tf.reshape(pool, shape=(-1, dim))
            if i == 0:
                conv = conv2d
            else:
                conv += conv2d  # tf.concat([conv, conv2d], axis=1)

        attention = Dense(concat.shape[1], activation='softmax', kernel_initializer=seed)(concat)
        attention = concat*attention

        res = tf.concat([attention, conv], axis=1)

        output = Dense(types, activation='softmax', kernel_initializer=seed)(res)

        input = [vi]
        if task == 'link':
            input = [vi, vj]
        model = Model(inputs=[input], outputs=[output])

        return model

    def train(self, task='classifier'):
        weights = self.init_embedding()
        data, label, types = None, None, None
        if task == 'classifier':
            data, label, types = self.graph.load_classes(self.classes_path) #cora:10,
            x_train, y_train, x_test, y_test = utils.train_test_split(data, label, rate=0.5, seed=10)
        if task == 'link':
            x_train, y_train = self.graph.sampled_link(num_neg=1)
            x_test, y_test = self.graph.sampled_link(num_neg=1, test=True)
            types = 2

        # y_preds[i]保存第i轮的测试集预测结果
        y_preds = np.zeros((len(x_test), self.folds))
        kf = KFold(n_splits=self.folds)
        for fold_n, (train_index, val_index) in enumerate(kf.split(y_train)):
            x_trn, y_trn = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]

            opt = Adam(0.01)
            model = self.create_model(weights, types, task=task)
            model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy())

            patient, best_score = 0, 0
            for epoch in range(100):
                generator = utils.batch_iter(x_trn, self.batch_size)
                for index in generator:
                    if task == 'classifier':
                        model.train_on_batch([x_trn[index]], np.eye(types)[y_trn[index]])
                    if task == 'link':
                        vi, vj = x_trn[index][:, 0], x_trn[index][:, 1]
                        model.train_on_batch([vi, vj], np.eye(types)[y_trn[index].reshape(-1).astype(int)])
                if task == 'classifier':
                    y_val_pred = np.argmax(model.predict([x_val]), -1)
                    micro, macro = Evaluate.f1(y_val, y_val_pred)
                    print('fold_{}:,{},{}'.format(fold_n, micro, macro))
                    score = micro+macro
                if task == 'link':
                    y_val_pred = np.argmax(model.predict([x_test[:, 0], x_test[:, 1]]), -1)
                    score = roc_auc_score(y_test, y_val_pred)
                    print('fold_{}:,{}'.format(fold_n, score), end='')

                if score > best_score:
                    patient = 0
                    best_score = score
                    model.save_weights('dataset/output/weights/classifier')
                print(',{}'.format(best_score))
                patient += 1
                if patient >= 30:
                    break

            model.load_weights('dataset/output/weights/classifier')
            if task == 'classifier':
                y_pred = model.predict(x_test)
            if task == 'link':
                y_pred = model.predict([x_test[:, 0], x_test[:, 1]])
            y_pred = np.argmax(y_pred, -1)  # 将概率最大所在的索引作为预测结果
            y_preds[:, fold_n] = y_pred
            score = Evaluate.f1(y_test, y_pred)
            print("fold_{}: {}".format(fold_n + 1, score))


        y_preds = stats.mode(y_preds, axis=1)[0].reshape(-1)
        if task == 'classifier':
            score = Evaluate.f1(y_test, y_preds)
            print('micro score: {}, macro score: {}'.format(score[0], score[1]))
        if task == 'link':
            score = roc_auc_score(y_test, y_preds)
            print('auc score: {}'.format(score))






