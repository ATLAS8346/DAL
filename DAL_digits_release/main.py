import argparse
import torch
import sys
import os
import numpy as np

from solver_disentangle import Solver
from pprint import pprint

sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric')


# Training settings
parser = argparse.ArgumentParser(description='PyTorch DAL Implementation')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_dir', type=str, default='./data/Digit-Five')
parser.add_argument('--eval_only', action='store_true', default=False, help='evaluation only option')

parser.add_argument('--max_epoch', type=int, default=150, metavar='N', help='how many epochs')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--num_k', type=int, default=4, metavar='N', help='hyper paremeter for generator update')

parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N', help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N', help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False, help='save_model or not')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')

parser.add_argument('--source', type=str, default='svhn', metavar='N', help='source dataset')
parser.add_argument('--target', type=str, default=None, metavar='N', help='target dataset')
parser.add_argument('--use_abs_diff', action='store_true', default=False, help='use absolute difference value as a measurement')

parser.add_argument('--exp_name', type=str, default='test', metavar='N')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Use cuda or not')
parser.add_argument('--opt_losses', nargs='+', type=str, default='', metavar='N', help='which losses')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

# Fix random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
pprint(args)

def main():
    solver = Solver(args)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if args.eval_only:
        solver.test(0)
    else:
        for epoch in range(args.max_epoch):
            solver.train_epoch(epoch)
            if epoch % 1 == 0:
                print("Epoch {}".format(epoch))
                print("==========================")
                solver.test(subset='train', save_model=args.save_model)
                solver.test(subset='test', save_model=args.save_model)
                print("==========================")


if __name__ == '__main__':
    main()
