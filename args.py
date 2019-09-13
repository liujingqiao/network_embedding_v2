import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Run Model.")
    emb = 'citeseer'
    data = emb+'/'+emb
    emb = 'citeseer/link'
    parser.add_argument('--edges_list', nargs='?', default='dataset/'+data+'.cites', help='Input graph path')
    parser.add_argument('--classes_input', nargs='?', default='dataset/'+data+'.classes', help='node category label path')
    parser.add_argument('--attr_input', nargs='?', default='dataset/'+data+'.attribute', help='attribute graph path')
    parser.add_argument('--attr_embedding', nargs="?", default='dataset/embedding/'+emb+'/attr_embedding.txt', help='embedding path')
    parser.add_argument('--stru_embedding', nargs="?", default='dataset/embedding/'+emb+'/stru_embedding.txt', help='struct path')
    parser.add_argument('--link_embedding', nargs="?", default='dataset/embedding/'+emb+'/link_embedding.txt', help='link path')
    parser.add_argument('--class_embedding', nargs="?", default='dataset/embedding/'+emb+'/class_embedding.txt', help='link path')
    parser.add_argument('--walk_embedding', nargs="?", default='dataset/embedding/'+emb+'/walk_embedding.txt', help='walk path')
    parser.add_argument('--walk_structure_embedding', nargs="?", default='dataset/embedding/'+emb+'/walk_struct_embedding.txt', help='walk path')
    parser.add_argument('--link_pred_embedding', nargs="?", default='dataset/embedding/'+emb+'/link_pred_embedding.txt', help='walk path')
    parser.add_argument('--walk_length', type=int, default=20, help='walk length')
    parser.add_argument('--folds', type=int, default=5, help='number of folds')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size.')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--emb_dim', type=int, default=256, help='Number of epochs.')
    parser.add_argument('--hidden_size', type=list, default=[512, 256], help='Number of epochs.')
    return parser.parse_args()


"""
cora
seed = RandomNormal(mean=0.0, stddev=0.05, seed=6)
x_train, y_train, x_test, y_test = utils.train_test_split(data, label, train_size=0.7, seed=42)
self.folds = 5

citeseer

"""
