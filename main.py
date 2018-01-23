from solver import Solver
import argparse

def main(config):
        solver = Solver(config)

        if config.mode == 'train':
            solver.train()
        else:
            solver.pretrain()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=32)

    config = parser.parse_args()
    main(config)