import networkx as nx
from graph import Graph
from args import parse_args
from pre_train import PreTraining
from interactive import Interactive



if __name__ == '__main__':
    args = parse_args()
    classes_path = args.classes_input
    edges_path = args.edges_list
    graph = Graph()
    graph.load_edgelist(edges_path)
    pre_train = PreTraining(graph, args)
    pre_train.walk_training(trained=False, repeat=100, walk_length=40)
    interactive = Interactive(graph, args)
    interactive.train()