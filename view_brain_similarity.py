import pickle
import argparse
from evaluate_brain_similarity import BENCHMARKS, LAYERS

CHECKPOINTS = ['alpha_0', 'alpha_0.00002', 'alpha_0.0001', 'alpha_0.0005', 'alpha_0.002', 'alpha_0.01', 'alpha_0.05']


def load_score(benchmark, checkpoint):
    with open(f'brain_score/{benchmark}/{checkpoint}.pkl', 'rb') as f:
        score_dict = pickle.load(f)
    return score_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View Brain Similarity.')
    parser.add_argument("--load-from", type=str, nargs='+', default=CHECKPOINTS, help='File to load from')
    args = parser.parse_args()

    for i in BENCHMARKS:
        print('\t'.join([i.split('.')[1].split('-')[0].ljust(15)] + LAYERS))
        for j in args.load_from:
            score_dict = load_score(i, j)
            print('\t'.join([j.ljust(15)] + [f'{float(score_dict[k]):.4f}±{float(score_dict[k].error):.4f}' for k in LAYERS]))
        print('')
