import numpy as np


def random_walk(adj, repeat=100, walk_length=40):
    """
    :param adj: 接收一个邻接矩阵
    :param repeat: 每个顶点重复游走的次数
    :param walk_length: 顶点每次游走的长度
    :return: 返回所有顶点游走的序列
    """
    walk_list = []
    for _ in range(repeat):
        for i in range(len(adj)):
            cur = i
            context = [cur]
            for _ in range(walk_length - 1):
                link_list = np.argwhere(adj[cur] == 1).reshape(-1)
                if len(link_list) > 0:
                    choose = np.random.randint(len(link_list))
                    context.append(link_list[choose])
                    cur = link_list[choose]
                else:
                    context.extend([len(adj)] * (walk_length - len(context)))
            walk_list.append(list(map(str, context)))
    return walk_list


def batch_iter(data, batch_size, shuffle=True, seed=None):
    """
    :param data: 要迭代的数据
    :param batch_size: 每次迭代得数量
    :param shuffle:  是否打乱数据顺序
    :param seed:  与shuffle配合使用
    :return:  返回batch_size数量的数据索引
    """
    data = np.array(data)
    data_size = len(data)
    index = np.arange(data_size)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(index)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield index[start_index: end_index]


def train_test_split(data, label, train_size=0.7, seed=None, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    index = np.arange(data_size)

    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(index)
    # index = np.random.permutation(np.arange(len(index)))

    data, label = data[index], label[index]
    train_size = int(data_size * train_size)
    x_train, y_train = data[0:train_size], label[0:train_size]
    x_test, y_test = data[train_size:], label[train_size:]

    return x_train, y_train, x_test, y_test
