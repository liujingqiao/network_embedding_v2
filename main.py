import networkx as nx
from args import parse_args



if __name__ == '__main__':
    args = parse_args()
    graph = nx.read_edgelist(args.train_input)
    model = MyModel(args, graph)
    model.train()
