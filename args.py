import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Run Model.")

    parser.add_argument('--edges_list', nargs='?', default='dataset/wiki/Wiki_edgelist.txt', help='Input graph path')
    parser.add_argument('--classes_input', nargs='?', default='dataset/wiki/Wiki_category.txt', help='node category label path')
    parser.add_argument('--attr_input', nargs='?', default='dataset/wiki/cora.attribute', help='attribute graph path')
    parser.add_argument('--attr_embedding', nargs="?", default='dataset/embedding/wiki/attr_embedding.txt', help='embedding path')
    parser.add_argument('--stru_embedding', nargs="?", default='dataset/embedding/wiki/stru_embedding.txt', help='struct path')
    parser.add_argument('--link_embedding', nargs="?", default='dataset/embedding/wiki/link_embedding.txt', help='link path')
    parser.add_argument('--walk_embedding', nargs="?", default='dataset/embedding/wiki/walk_embedding.txt', help='walk path')
    parser.add_argument('--walk_length', type=int, default=20, help='walk length')
    parser.add_argument('--folds', type=int, default=5, help='number of folds')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size.')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--emb_dim', type=int, default=256, help='Number of epochs.')
    parser.add_argument('--hidden_size', type=list, default=[512, 256], help='Number of epochs.')
    return parser.parse_args()

