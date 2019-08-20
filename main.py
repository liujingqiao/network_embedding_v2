from graph import Graph
from args import parse_args
from pre_train import PreTraining
from interactive import Interactive


def train(walk=False, struc=False, link=False, attr=False, walk_struc=False):
    pre_train = PreTraining(graph, args)
    pre_train.walk_proximity(trained=walk, num_walks=80, walk_length=20, p=1, q=1)
    pre_train.structure_proximity(trained=struc)
    pre_train.link_proximity(trained=link)
    pre_train.attributes_proximity(trained=attr)
    pre_train.walk_structure_proximity(trained=walk_struc, num_walks=10, walk_length=5, p=1, q=1)

if __name__ == '__main__':
    args = parse_args()
    classes_path = args.classes_input
    edges_path = args.edges_list
    graph = Graph()
    #graph.load_edgelist(edges_path)
    train(walk=True, struc=True, link=True, attr=True, walk_struc=True)
    interactive = Interactive(graph, args)
    interactive.train()
