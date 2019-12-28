import sys
import numpy as np
sys.path.append('../loader')
from .unaligned_data_loader import UnalignedDataLoader
from .svhn import load_svhn
from .mnist import load_mnist
from .mnist_m import load_mnistm
from .usps_ import load_usps
from .synth_number import load_syn


def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    if data == 'usps':
        train_image, train_label, test_image, test_label = load_usps(all_use=all_use)
    if data == 'mnistm':
        train_image, train_label, test_image, test_label = load_mnistm()
    if data == 'synth':
        train_image, train_label, test_image, test_label = load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, test_image, test_label = load_gtsrb()
    if data == 'syn':
        train_image, train_label, test_image, test_label = load_syn()

    return train_image, train_label, test_image, test_label


def dataset_read(source, target, batch_size, scale=False, all_use='no'):

    train_src, test_src = {}, {}
    train_trg, test_trg = {}, {}

    usps = True if source == 'usps' or target == 'usps' else False
    # domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
    domain_all = ['mnistm', 'mnist', 'usps', 'svhn']
    domain_all.remove(source)
    (train_src_img, train_src_label,
     test_src_img, test_src_label) = return_dataset(
         source, scale=scale, usps=usps, all_use=all_use)

    train_trg_img, train_trg_label = [], []
    test_trg_img, test_trg_label = [], []
    for i in range(len(domain_all)):
        (_train_img, _train_label, _test_img, _test_label) = return_dataset(
            domain_all[i], scale=scale, usps=usps, all_use=all_use)
        train_trg_img.append(_train_img)
        train_trg_label.append(_train_label)
        test_trg_img.append(_test_img)
        test_trg_label.append(_test_label)

    train_trg_img = np.concatenate(train_trg_img, axis=0)
    train_trg_label = np.concatenate(train_trg_label, axis=0)
    test_trg_img = np.concatenate(test_trg_img, axis=0)
    test_trg_label = np.concatenate(test_trg_label, axis=0)

    # print(domain)
    print('Source Training: ', train_src_img.shape)
    print('Source Training label: ', train_src_label.shape)
    print('Source Test: ', test_src_img.shape)
    print('Source Test label: ', test_src_label.shape)

    print('Target Training: ', train_trg_img.shape)
    print('Target Training label: ', train_trg_label.shape)
    print('Target Test: ', test_trg_img.shape)
    print('Target Test label: ', test_trg_label.shape)

    train_src['imgs'] = train_src_img
    train_src['labels'] = train_src_label
    train_trg['imgs'] = train_trg_img
    train_trg['labels'] = train_trg_label

    # input target samples for both
    test_src['imgs'] = test_src_img
    test_src['labels'] = test_src_label
    test_trg['imgs'] = test_trg_img
    test_trg['labels'] = test_trg_label
    scale = 32 if source == 'synth' else 32 if source == 'usps' or target == 'usps' else 32

    train_loader = UnalignedDataLoader()
    train_loader.initialize(train_src, train_trg, batch_size, batch_size, scale=scale)
    dataset_train = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(test_src, test_trg, batch_size, batch_size, scale=scale)
    dataset_test = test_loader.load_data()

    return dataset_train, dataset_test
