import numpy as np
from scipy.io import loadmat
import sys

sys.path.append('../utils/')
from utils.utils import dense_to_one_hot


def load_syn(base_dir, scale=32):
    syn_train = loadmat(base_dir + '/synth_train_{}x{}.mat'.format(scale, scale))
    syn_test = loadmat(base_dir + '/synth_test_{}x{}.mat'.format(scale, scale))

    # train
    syn_train_im = syn_train['X']
    syn_train_im = syn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = dense_to_one_hot(syn_train['y'].squeeze())
    # idx = np.random.permutation(syn_train_im.shape[0])
    # syn_train_im = syn_train_im[idx]
    # train_label = train_label[idx]
    # syn_train_im = syn_train_im[:25000]
    # train_label = train_label[:25000]

    # test
    syn_test_im = syn_test['X']
    syn_test_im = syn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    test_label = dense_to_one_hot(syn_test['y'].squeeze())
    # syn_test_im = syn_test_im[:9000]
    # test_label = test_label[:9000]

    print('syn number train X shape->', syn_train_im.shape)
    print('syn number train y shape->', train_label.shape)
    print('syn number test X shape->', syn_test_im.shape)
    print('syn number test y shape->', test_label.shape)
    print("====================")
    return syn_train_im, train_label, syn_test_im, test_label



