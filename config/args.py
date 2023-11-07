import argparse

def get_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--dataName', type=str, default="ml-100k")
    parser.add_argument('--model-path', type=str, default="./models")
    parser.add_argument('--data-path', type=str, default="./data")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=512)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--init', type=str)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--drop', type=float, default=0.7)

    parser.add_argument('--emb-dim', type=int, default=32)
    parser.add_argument('--hidden', default=[64,32,16, 8])
    parser.add_argument('--nb', type=int, default=2)

    parser.add_argument('--train-path', type=str, default='./data/rating_train.pkl')
    parser.add_argument('--val-path', type=str, default='./data/rating_val.pkl')
    parser.add_argument('--test-path', type=str, default='./data/rating_test.pkl')

    parser.add_argument('--pop_thr', type=float, default=0.05)
    parser.add_argument('--n_ran', type=int, default=5)
    parser.add_argument('--ratio', type=float, default=5)
    parser.add_argument('--recommender', type=str, default='WMF')
    parser.add_argument('--limit', type=int, default='limit')
    parser.add_argument("--target", type=int, default=1000, help="target to attack")
    parser.add_argument("--budget", type=int, default=50, help='attack budget')
    parser.add_argument("--attack_size", type=int, default=50, help='attack size')
    parser.add_argument("--mode", type=bool, default=True, help='attack size')


    args = parser.parse_args()

    return args