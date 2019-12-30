import sys
import numpy as np
sys.path.append('../loader')
from .unaligned_data_loader import UnalignedDataLoader
from .svhn import load_svhn
from .mnist import load_mnist
from .mnist_m import load_mnistm
from .usps_ import load_usps
from .synth_number import load_syn
# from .synth_traffic import load_syntraffic
# from .gtsrb import load_gtsrb


def return_dataset(data, base_dir, scale=28):

    if data == 'svhn':
        return load_svhn(base_dir, scale=scale)

    elif data == 'mnist':
        return load_mnist(base_dir, scale=scale)

    elif data == 'usps':
        return load_usps(base_dir)

    elif data == 'mnistm':
        return load_mnistm(base_dir)

    elif data == 'syn':
        return load_syn(base_dir, scale=scale)

    # elif data == 'synth':
    #     return load_syntraffic(base_dir)

    # elif data == 'gtsrb':
    #     return load_gtsrb(base_dir)

    else:
        raise NotImplementedError("Dataset not found")


def dataset_read(base_dir, source, target, batch_size, scale=32):

    train_src, test_src = {}, {}
    train_trg, test_trg = {}, {}

    if target is None:
        domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
        domain_all.remove(source)
    else:
        domain_all = [target]

    # NOTE: scale here, is first fixed to 28 to retrieve images of the same size
    # scale=32 is used byt the dataloader to rescale images at train/test time

    # get source domain
    (train_src_img, train_src_label,
     test_src_img, test_src_label) = return_dataset(
         source, base_dir, scale=28)

    # get target domains
    train_trg_img, train_trg_label = [], []
    test_trg_img, test_trg_label = [], []
    for i in range(len(domain_all)):
        (_train_img, _train_label, _test_img, _test_label) = return_dataset(
            domain_all[i], base_dir, scale=28)

        train_trg_img.append(_train_img)
        train_trg_label.append(_train_label)
        test_trg_img.append(_test_img)
        test_trg_label.append(_test_label)

    # aggregate all domains
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

    train_loader = UnalignedDataLoader(
        train_src, train_trg, batch_size, batch_size, scale=scale)
    dataset_train = train_loader.load_data()
    print("Train batches: ", len(train_loader))

    test_loader = UnalignedDataLoader(
        test_src, test_trg, batch_size, batch_size, scale=scale)
    dataset_test = test_loader.load_data()
    print("Test batches: ", len(test_loader))

    return dataset_train, dataset_test
