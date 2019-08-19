import numpy as np
import multiprocessing


def random_walk(start, end, adj, repeat=100, walk_length=40):
    """
    :param adj: 接收一个邻接矩阵
    :param repeat: 每个顶点重复游走的次数
    :param walk_length: 顶点每次游走的长度
    :return: 返回所有顶点游走的序列
    """
    walk_list = []
    for _ in range(repeat):
        for i in range(start, end):
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


def random_walk_parallel(adj, repeat=100, walk_length=40, num_proceeding=10):
    proceeding = []
    num_nodes = int(len(adj)/num_proceeding)
    block = []
    for i in range(num_proceeding - 1):
        block.append((int(i*num_nodes), int((i+1)*num_nodes)))
    block.append((int((num_proceeding-1)*num_nodes), len(adj)))
    pool = multiprocessing.Pool(processes=num_proceeding)
    for i, (start, end) in enumerate(block):
        proceeding.append(pool.apply_async(random_walk, (start, end, adj, repeat, walk_length)))
    pool.close()
    pool.join()
    corpus = []
    for p in proceeding:
        corpus.extend(p.get())
    return corpus



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


def create_alias_table(area_ratio):
    """

    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]#小概率的值保存到accept中
        alias[small_idx] = large_idx#大概率的索引保存到alias中
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
            (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """

    :param accept:
    :param alias:
    :return: sample index
    """
    # 根据概率选择游走的邻居
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
