import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Run Model.")

    parser.add_argument('--train_input', nargs='?', default='../data/wiki/Wiki_edgelist.txt', help='Input graph path')
    parser.add_argument('--classes_input', nargs='?', default='../data/wiki/Wiki_category.txt', help='node category label path')
    parser.add_argument('--attr_input', nargs='?', default='../data/wiki/citeseer.attribute', help='attribute graph path')
    parser.add_argument('--attr_output', nargs="?", default='../data/output/wiki/attr_embedding.txt', help='embedding path')
    parser.add_argument('--stru_output', nargs="?", default='../data/output/wiki/stru_embedding.txt', help='struct path')
    parser.add_argument('--link_output', nargs="?", default='../data/output/wiki/link_embedding.txt', help='link path')
    parser.add_argument('--walk_output', nargs="?", default='../data/output/wiki/walk_embedding.txt', help='walk path')
    parser.add_argument('--walk_length', type=int, default=20, help='walk length')
    parser.add_argument('--folds', type=int, default=5, help='number of folds')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size.')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--attr_hidden', type=list, default=[512, 256], help='Number of epochs.')
    parser.add_argument('--stru_hidden', type=list, default=[512, 256], help='Number of epochs.')
    return parser.parse_args()

