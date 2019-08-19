import numpy as np


class Graph(dict):
    def __init__(self, is_directed=False):
        dict.__init__(self)
        self.is_directed = is_directed
        self.edges = None
        self.nodes = None
        self.attributes = None
        self.node2idx = None
        self.idx2node = None
        self.adj = None

    def _indexed_node(self):
        # indexed node
        node2idx, idx2node = {}, {}
        for i, node in enumerate(self.nodes):
            if node not in node2idx:
                node2idx[node] = i
                idx2node[i] = node
                i = i + 1
        self.node2idx = node2idx
        self.idx2node = idx2node
        self.nodes = np.array([node2idx[node] for node in self.nodes])

    def load_edgelist(self, filename):
        nodes = []
        with open(filename, 'r') as fr:
            for line in fr:
                line = line.split()
                # 构建图
                if line[0] not in self:
                    self[line[0]] = {}
                if line[1] not in self:
                    self[line[1]] = {}
                self[line[0]][line[1]] = 1
                if not self.is_directed:
                    self[line[1]][line[0]] = 1
                # 保存顶点集合
                if line[0] not in nodes:
                    nodes.append(line[0])
                if line[1] not in nodes:
                    nodes.append(line[1])
        self.nodes = nodes
        self._indexed_node()
        self._make_adj_edges()
        return self

    def load_classes(self, path):
        classes2idx = {}
        node, classes = [], []
        with open(path, 'r') as fr:
            for line in fr:
                line = line.split()
                classes2idx[line[1]] = classes2idx.get(line[1], len(classes2idx))
                node.append(self.node2idx[line[0]])
                classes.append(classes2idx[line[1]])
        types = len(set(classes))
        return np.array(node), np.array(classes), types

    def _make_adj_edges(self):
        node2idx = self.node2idx
        node_num = len(self.node2idx)
        edges = []
        adj = np.zeros((node_num, node_num))
        for vi, link in self.items():
            for vj, weight in link.items():
                i, j = node2idx[vi], node2idx[vj]
                adj[i][j] = 1
                edges.append([i, j])
        self.adj = adj
        self.edges = edges
        return adj

    def neighbors(self, node):
        return np.argwhere(self.adj[node] == 1).reshape(-1)

    def load_attribute(self, filename, types=1):
        # 数据格式不同处理方式不同
        if types == 1:
            with open(filename, "r") as fr:
                line = fr.readline().split()
                dim = len(line[1:])
            attrs = np.zeros((len(self.node2idx), dim))
            with open(filename, 'r') as fr:
                for line in fr:
                    line = line.split()
                    node, attr = self.node2idx[line[0]], list(map(int, line[1:]))
                    attrs[node] = attr
            self.attributes = attrs
            return attrs
        if type == 2:
            node_attr, attr2idx, i = {}, {}, 0
            with open(filename, 'r') as fr:
                for line in fr:
                    line = line.split()
                    for attr in line[1:]:
                        if attr not in attr2idx:
                            attr2idx[attr] = i
                            i += 1
                    node_attr[self.node2idx[line[0]]] = [attr2idx[attr] for attr in line[1:]]

            node_attr_matrix = np.zeros((len(self.node2idx), len(attr2idx)))
            for item in node_attr.items():
                for attr in item[0]:
                    node_attr_matrix[item[0]][attr] = 1
            self.attributes = node_attr_matrix
            return node_attr_matrix

    def sampled_link(self, num_neg=5):
        # 正样本
        links_list = self.edges
        label = np.ones((len(links_list),1))
        # 负采样
        for i in range(len(self.edges)):
            vi = links_list[i][0]
            # 每个正样本对应采样sample_rate个负样本
            for j in range(num_neg):
                nag = np.argwhere(self.adj[vi] == 0).reshape(-1)
                if len(nag) == 0:
                    break
                vj = nag[np.random.randint(len(nag))]
                self.adj[vi][vj] = 1
                links_list = np.append(links_list, [[vi, vj]], axis=0)
                label = np.append(label, [[-1]], axis=0)
        shuffle_index = np.random.permutation(np.arange(len(links_list)))
        links_list = links_list[shuffle_index]
        label = label[shuffle_index]
        return links_list, label

    def has_edge(self, vi, vj):
        if self.adj[vi][vj] == 1:
            return True
        return False

