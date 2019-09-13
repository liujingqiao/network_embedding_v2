import numpy as np
import multiprocessing


def deep_walk(adj, num_walks=100, walk_length=40, workers=None):
    """
    :param adj: 接收一个邻接矩阵
    :param num_walks: 每个顶点重复游走的次数
    :param walk_length: 顶点每次游走的长度
    :return: 返回所有顶点游走的序列
    """
    walk_list = []
    for _ in range(num_walks):
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


def _deep_walk(adj, num_walks=100, walk_length=10, workers=5):
    proceeding = []
    num_nodes = int(len(adj)/workers)
    block = []
    for i in range(workers - 1):
        block.append((int(i*num_nodes), int((i+1)*num_nodes)))
    block.append((int((workers-1)*num_nodes), len(adj)))
    pool = multiprocessing.Pool(processes=workers)
    for i, (start, end) in enumerate(block):
        proceeding.append(pool.apply_async(_deep_walk, (start, end, adj, num_walks, walk_length)))
    pool.close()
    pool.join()
    corpus = []
    for p in proceeding:
        corpus.extend(p.get())
    return corpus


def walk_proximity(adj, num_walks=100, walk_length=10, workers=5):
    walks = deep_walk(adj, num_walks=num_walks, walk_length=walk_length, workers=workers)
    walk_structure = np.zeros_like(adj)
    for walk in walks:
        try:
            walk = list(map(int, walk))
            walk_structure[walk[0]][walk[1:]] = 1
        except:
            print(walk[0],walk[1:])
    return np.array(walk_structure)


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


def train_test_split(data, label, rate=0.5, seed=None):
    label, data = np.array(label), np.array(data)

    classes = np.unique(label)
    train_index, test_index = [], []

    for types in classes:
        idx = np.argwhere(label == types)[:, 0]
        if seed:
            np.random.seed(seed)
        np.random.shuffle(idx)

        train_size = int(len(idx) * rate) + 1
        train_index.extend(list(idx[:train_size]))
        test_index.extend(list(idx[train_size:]))
    if seed:
        np.random.seed(seed)
    shuffle_index = np.random.permutation(np.arange(len(label)))
    data, label = data[shuffle_index], label[shuffle_index]
    x_train, y_train = data[train_index], label[train_index]
    x_test, y_test = data[test_index], label[test_index]

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


def pubmed_cites():
    fw = open('dataset/pubmed/pubmed.cites', 'w')
    with open('dataset/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab') as fr:
        lines = fr.readlines()[2:]
        for line in lines:
            line = line.strip().split()
            vi, vj = line[1].split(':')[1], line[3].split(':')[1]
            fw.write(vi+' '+vj+'\n')
        fw.close()


def pubmed_classes():
    fw = open('dataset/pubmed/pubmed.classes', 'w')
    with open('dataset/pubmed/Pubmed-Diabetes.NODE.paper.tab') as fr:
        lines = fr.readlines()[2:]
        for line in lines:
            line = line.strip().split()
            node, label = line[0], line[1][-1]
            fw.write(node+' '+label+'\n')
        fw.close()


def pubmed_attributes():
    attr2id = dict()
    fw = open('dataset/pubmed/pubmed.attribute', 'w')
    with open('dataset/pubmed/Pubmed-Diabetes.NODE.paper.tab') as fr:
        lines = fr.readlines()
        attrs = lines[1].strip().split()[1:]
        for attr in attrs:
            attr = attr.split(':')[1]
            if attr not in attr2id:
                attr2id[attr] = len(attr2id)
        for line in lines[2:]:
            line = line.split()
            node = line[0]
            vector = np.zeros(len(attr2id))
            for attr in line[2:]:
                attr = attr.split('=')[0]
                vector[attr2id[attr]] = 1
            vector = node+' '+' '.join([str(int(i)) for i in vector])+'\n'
            fw.write(vector)

        fw.close()


# 为图文件划分训练集和测试集
def split_test_file(file, rate=0.5):
    path = 'dataset/'+file+'/'+file+'.cites'
    train_edges = []
    with open(path, 'r') as fr:
        for line in fr:
            train_edges.append(line)
    np.random.shuffle(train_edges)
    train_size = int(len(train_edges)*rate)
    test_edges, train_edges = train_edges[:train_size], train_edges[train_size:]
    path = 'dataset/'+file+'/'+file+'_train.cites'
    with open(path, 'w') as fw:
        for edges in train_edges:
            fw.write(edges)
    path = 'dataset/'+file+'/'+file+'_test.citess'
    with open(path, 'w') as fw:
        for edges in test_edges:
            fw.write(edges)




