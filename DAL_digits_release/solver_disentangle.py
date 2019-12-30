from __future__ import print_function
import torch
import sys
import os
sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric')
print(sys.path)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.build_gen import Disentangler, Generator, Classifier, Feature_Discriminator, Reconstructor, Mine
from datasets.dataset_read import dataset_read
from utils.utils import _l2_rec, _ent, _discrepancy, _ring

from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from tqdm import tqdm

class Solver(nn.Module):
    def __init__(self, args):

        super().__init__()

        timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_%s" % args.exp_name
        self.logdir = os.path.join('./logs', timestring)
        self.logger = SummaryWriter(log_dir=self.logdir)
        self.device = torch.device("cuda" if args.use_cuda else "cpu")

        self.src_domain_code = np.repeat(
            np.array([[*([1]), *([0])]]), args.batch_size, axis=0)
        self.trg_domain_code = np.repeat(
            np.array([[*([0]), *([1])]]), args.batch_size, axis=0)
        self.src_domain_code = torch.FloatTensor(
            self.src_domain_code).to(self.device)
        self.trg_domain_code = torch.FloatTensor(
            self.trg_domain_code).to(self.device)

        self.source = args.source
        self.target = args.target
        self.num_k = args.num_k
        self.checkpoint_dir = args.checkpoint_dir
        self.save_epoch = args.save_epoch
        self.use_abs_diff = args.use_abs_diff

        self.mi_k = 1
        self.delta = 0.01
        self.mi_coeff = 0.0001
        self.interval = 10  # write on tb every
        self.batch_size = args.batch_size
        self.which_opt = 'adam'
        self.lr = args.lr
        self.scale = 32
        self.global_step = 0

        print('Loading datasets')
        self.dataset_train, self.dataset_test = dataset_read(
            args.data_dir, self.source, self.target,
            self.batch_size, self.scale)
        print('Done!')

        self.total_batches = {
            'train': self.get_dataset_size('train'),
            'test': self.get_dataset_size('test')
        }

        self.G = Generator(source=self.source, target=self.target)
        self.FD = Feature_Discriminator()
        self.R = Reconstructor()
        self.MI = Mine()

        self.C = nn.ModuleDict({
            'ds': Classifier(source=self.source, target=self.target),
            'di': Classifier(source=self.source, target=self.target),
            'ci': Classifier(source=self.source, target=self.target)
        })

        self.D = nn.ModuleDict({
            'ds': Disentangler(), 'di': Disentangler(), 'ci': Disentangler()})

        # All modules in the same dict
        self.components = nn.ModuleDict({
            'G': self.G, 'FD': self.FD, 'R': self.R, 'MI': self.MI
        })

        self.xent_loss = nn.CrossEntropyLoss().cuda()
        self.adv_loss = nn.BCEWithLogitsLoss().cuda()
        self.set_optimizer(which_opt=self.which_opt, lr=self.lr)
        self.to_device()

    def get_dataset_size(self, subset):
        dataset = self.dataset_test if subset == 'test' else self.dataset_train
        count = 0
        for batch_idx, data in enumerate(dataset):
            img_src, img_trg = data['S'], data['T']
            if (img_src.size()[0] < self.batch_size or
                    img_trg.size()[0] < self.batch_size):
                continue
            count += 1
        return count

    def to_device(self):
        for k, v in self.components.items():
            self.components[k] = v.cuda()

        for k, v in self.C.items():
            self.C[k] = v.cuda()

        for k, v in self.D.items():
            self.D[k] = v.cuda()

    def set_optimizer(self, which_opt='adam', lr=0.001, momentum=0.9):
        self.opt = {
            'C_ds': optim.Adam(self.C['ds'].parameters(), lr=lr, weight_decay=5e-4),
            'C_di': optim.Adam(self.C['di'].parameters(), lr=lr, weight_decay=5e-4),
            'C_ci': optim.Adam(self.C['ci'].parameters(), lr=lr, weight_decay=5e-4),
            'D_ds': optim.Adam(self.D['ds'].parameters(), lr=lr, weight_decay=5e-4),
            'D_di': optim.Adam(self.D['di'].parameters(), lr=lr, weight_decay=5e-4),
            'D_ci': optim.Adam(self.D['ci'].parameters(), lr=lr, weight_decay=5e-4),
            'G': optim.Adam(self.G.parameters(), lr=lr, weight_decay=5e-4),
            'FD': optim.Adam(self.FD.parameters(), lr=lr, weight_decay=5e-4),
            'R': optim.Adam(self.R.parameters(), lr=lr, weight_decay=5e-4),
            'MI': optim.Adam(self.MI.parameters(), lr=lr, weight_decay=5e-4),
        }

    def reset_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()

    def mi_estimator(self, x, y, y_):
        joint, marginal = self.MI(x, y), self.MI(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def group_opt_step(self, opt_keys):
        for k in opt_keys:
            self.opt[k].step()
        self.reset_grad()

    def log_scalar(self, _loss, prefix=''):
        if self.global_step % self.interval == 0:
            for k, val in _loss.items():
                self.logger.add_scalar(
                    "%s/%s" % (prefix, k), val,
                    global_step=self.global_step)

    def optimize_classifier(self, img_src, label_src):
        _loss = dict()
        _acc = dict()
        feat_src = self.G(img_src)
        for key in ['ds', 'di', 'ci']:
            k = "class_src_%s" % key
            out = self.C[key](self.D[key](feat_src))
            _loss[k] = self.xent_loss(out, label_src)
            _acc[k] = (out.detach().max(1)[1] == label_src).float().sum() / out.size(0)

        self.log_scalar(_loss, prefix='class_loss')
        self.log_scalar(_acc, prefix='train_acc')

        _sum_loss = sum([l for _, l in _loss.items()])
        _sum_loss.backward()
        self.group_opt_step(['G', 'C_ds', 'C_di', 'C_ci', 'D_ds', 'D_di', 'D_ci'])
        return _loss

    def discrepancy_minimizer(self, img_src, img_trg, label_src):
        _loss = dict()
        # NOTE: I'm still not sure why we need this
        # on source domain
        feat_src = self.G(img_src)
        _loss['ds_src'] = self.xent_loss(
            self.C['ds'](self.D['ds'](feat_src)), label_src)
        _loss['di_src'] = self.xent_loss(
            self.C['di'](self.D['di'](feat_src)), label_src)

        # on target domain
        feat_trg = self.G(img_trg)
        _loss['discrepancy_ds_di_trg'] = _discrepancy(
            self.C['ds'](self.D['ds'](feat_trg)),
            self.C['di'](self.D['di'](feat_trg)))

        self.log_scalar(_loss, prefix='dis_loss')
        _sum_loss = sum([l for _, l in _loss.items()])
        _sum_loss.backward()
        self.group_opt_step(['D_ds', 'D_di', 'C_ds', 'C_di'])
        return _loss

    def ring_loss_minimizer(self, img_src, img_trg):
        data = torch.cat((img_src, img_trg), 0)
        feat = self.G(data)
        ring_loss = _ring(feat)
        ring_loss.backward()
        self.group_opt_step(['G'])
        self.log_scalar({'ring': ring_loss}, 'extra_loss')

        return ring_loss

    def mutual_information_minimizer(self, img_src, img_trg):
        # minimize mutual information between (d0, d2) and (d1, d2)
        # minimize mutual information between (ds, di) and (ci, di)

        for i in range(0, self.mi_k):
            feat_src, feat_trg = self.G(img_src), self.G(img_trg)
            ds_src, ds_trg = self.D['ds'](feat_src), self.D['ds'](feat_trg)
            di_src, di_trg = self.D['di'](feat_src), self.D['di'](feat_trg)
            ci_src, ci_trg = self.D['ci'](feat_src), self.D['ci'](feat_trg)

            di_src_shuffle = torch.index_select(
                di_src, 0, torch.randperm(di_src.shape[0]).to(self.device))
            di_trg_shuffle = torch.index_select(
                di_trg, 0, torch.randperm(di_trg.shape[0]).to(self.device))

            MI_ds_di_src = self.mi_estimator(ds_src, di_src, di_src_shuffle)
            MI_ds_di_trg = self.mi_estimator(ds_trg, di_trg, di_trg_shuffle)
            MI_ds_di = MI_ds_di_src + MI_ds_di_trg

            MI_ci_di_src = self.mi_estimator(ci_src, di_src, di_src_shuffle)
            MI_ci_di_trg = self.mi_estimator(ci_trg, di_trg, di_trg_shuffle)
            MI_ci_di = MI_ci_di_src + MI_ci_di_trg

            MI = 0.25 * (MI_ds_di_src + MI_ds_di_trg + MI_ci_di_src + MI_ci_di_trg) * self.mi_coeff
            MI.backward()
            self.group_opt_step(['D_ds', 'D_di', 'D_ci', 'MI'])

        self.log_scalar({'ds_di': MI_ds_di, 'ci_di': MI_ci_di}, prefix='MI')
        # pred_di_ci_src = self.M(out_di_src, out_ci_src)
        return MI_ds_di, MI_ci_di

    def class_confusion(self, img_src, img_trg):
        # - adversarial training

        # f_ci = CI(G(im)) extracts features that are class irrelevant
        # by maximizing the entropy, given that the classifier is fixed
        _loss = dict()
        feat_src = self.G(img_src)
        feat_trg = self.G(img_trg)
        _loss['src_ci'] = _ent(self.C['ci'](self.D['ci'](feat_src)))
        _loss['trg_ci'] = _ent(self.C['ci'](self.D['ci'](feat_trg)))
        self.log_scalar(_loss, prefix='confusion_loss')

        _sum_loss = sum([l for _, l in _loss.items()])
        _sum_loss.backward()
        self.group_opt_step(['D_ci', 'G'])
        return _loss

    def adversarial_alignment(self, img_src, img_trg):

        # FD should guess if the features extracted f_di = DI(G(im))
        # are from target or source domain. To win this game and fool FD,
        # DI should extract domain invariant features.

        # Loss measures features' ability to fool the discriminator
        src_domain_pred = self.FD(self.D['di'](self.G(img_src)))
        tgt_domain_pred = self.FD(self.D['di'](self.G(img_trg)))
        df_loss_src = self.adv_loss(src_domain_pred, self.src_domain_code)
        df_loss_trg = self.adv_loss(tgt_domain_pred, self.trg_domain_code)
        alignment_loss1 = 0.01 * (df_loss_src + df_loss_trg)
        alignment_loss1.backward()
        self.group_opt_step(['FD', 'D_di', 'G'])

        # Measure discriminator's ability to classify source from target samples
        src_domain_pred = self.FD(self.D['di'](self.G(img_src)))
        tgt_domain_pred = self.FD(self.D['di'](self.G(img_trg)))
        df_loss_src = self.adv_loss(src_domain_pred, 1 - self.src_domain_code)
        df_loss_trg = self.adv_loss(tgt_domain_pred, 1 - self.trg_domain_code)
        alignment_loss2 = 0.01 * (df_loss_src + df_loss_trg)
        alignment_loss2.backward()
        self.group_opt_step(['FD', 'D_di', 'G'])

        for _ in range(self.num_k):
            feat_trg = self.G(img_trg)
            loss_dis = _discrepancy(
                self.C['ds'](self.D['ds'](feat_trg)),
                self.C['di'](self.D['di'](feat_trg)))
            loss_dis.backward()
            self.group_opt_step(['G'])

        self.log_scalar(
            {'alignment_loss1': alignment_loss1,
             'alignment_loss2': alignment_loss2,
             'discrepancy_ds_di_trg': loss_dis},
            prefix='adv_loss')

        return alignment_loss1, alignment_loss2, loss_dis

    def optimize_rec(self, img_src, img_trg):
        _feat_src = self.G(img_src)
        _feat_trg = self.G(img_trg)

        feat_src, feat_trg = dict(), dict()
        rec_src, rec_trg = dict(), dict()
        for k in ['ds', 'di', 'ci']:
            feat_src[k] = self.D[k](_feat_src)
            feat_trg[k] = self.D[k](_feat_trg)

        recon_loss = None
        rec_loss_src, rec_loss_trg = dict(), dict()
        for k1, k2 in [('di', 'ci'), ('di', 'ds')]:
            k = '%s_%s' % (k1, k2)
            rec_src[k] = self.R(torch.cat([feat_src[k1], feat_src[k2]], 1))
            rec_trg[k] = self.R(torch.cat([feat_trg[k1], feat_trg[k2]], 1))
            rec_loss_src[k] = _l2_rec(rec_src[k], _feat_src)
            rec_loss_trg[k] = _l2_rec(rec_trg[k], _feat_trg)

            if recon_loss is None:
                recon_loss = rec_loss_src[k] + rec_loss_trg[k]
            else:
                recon_loss += rec_loss_src[k] + rec_loss_trg[k]

        self.log_scalar(rec_loss_src, prefix='rec_loss_src')
        self.log_scalar(rec_loss_trg, prefix='rec_loss_trg')

        recon_loss = (recon_loss / 4) * self.delta
        recon_loss.backward()
        self.group_opt_step(['D_di', 'D_ci', 'D_ds', 'R'])
        return rec_loss_src, rec_loss_trg

    def train_epoch(self, epoch):
        # set training
        self.train()

        total_batches = self.total_batches['train']
        pbar_descr_prefix = "Epoch %d" % (epoch)
        with tqdm(total=total_batches, ncols=80, dynamic_ncols=False,
                  desc=pbar_descr_prefix) as pbar:

            for batch_idx, data in enumerate(self.dataset_train):
                img_trg = data['T'].to(self.device)
                img_src = data['S'].to(self.device)
                label_src = data['S_label'].long().to(self.device)

                if (img_src.size()[0] < self.batch_size or
                    img_trg.size()[0] < self.batch_size):
                    continue

                self.reset_grad()
                # ================================== #
                self.optimize_classifier(img_src, label_src)
                self.discrepancy_minimizer(img_src, img_trg, label_src)
                self.ring_loss_minimizer(img_src, img_trg)
                self.mutual_information_minimizer(img_src, img_trg)
                self.class_confusion(img_src, img_trg)
                self.adversarial_alignment(img_src, img_trg)
                self.optimize_rec(img_src, img_trg)
                # ================================== #
                pbar.update()
                self.global_step += 1

        return self.global_step

    def test(self, subset='test', save_model=False):
        self.eval()

        dataset = self.dataset_test if subset == 'test' else self.dataset_train
        with torch.no_grad():
            loss_src, loss_trg = dict(), dict()
            correct_src, correct_trg = dict(), dict()
            size_src, size_trg = 0, 0
            for key in ['di', 'ds', 'ci']:
                loss_src[key], loss_trg[key] = 0, 0
                correct_src[key], correct_trg[key] = 0, 0

            pbar_descr_prefix = "Eval %s" % (subset)
            with tqdm(total=self.total_batches[subset], ncols=80, dynamic_ncols=False,
                      desc=pbar_descr_prefix) as pbar:

                for batch_idx, data in enumerate(dataset):
                    img_src, label_src = data['S'], data['S_label'].long()
                    img_src, label_src = img_src.to(self.device), label_src.to(self.device)

                    img_trg, label_trg = data['T'], data['T_label'].long()
                    img_trg, label_trg = img_trg.to(self.device), label_trg.to(self.device)

                    out_src, out_trg = dict(), dict()
                    pred_src, pred_trg = dict(), dict()

                    feat_src = self.G(img_src)
                    feat_trg = self.G(img_trg)
                    for key in ['di', 'ds', 'ci']:
                        out_src[key] = self.C[key](self.D[key](feat_src))
                        out_trg[key] = self.C[key](self.D[key](feat_trg))
                        # preds
                        pred_src[key] = F.softmax(out_src[key], dim=1).max(1)[1]
                        pred_trg[key] = F.softmax(out_trg[key], dim=1).max(1)[1]
                        # losses
                        loss_src[key] += F.cross_entropy(out_src[key], label_src, reduction='sum').item()
                        loss_trg[key] += F.cross_entropy(out_trg[key], label_trg, reduction='sum').item()
                        # correct predictions
                        correct_src[key] += pred_src[key].eq(label_src).cpu().sum()
                        correct_trg[key] += pred_trg[key].eq(label_trg).cpu().sum()

                    size_src += label_src.data.size()[0]
                    size_trg += label_trg.data.size()[0]

                    pbar.update(1)

        print("Source - {}".format(subset))
        acc, loss = dict(), dict()
        for key in ['di', 'ds', 'ci']:
            acc[key] = float(correct_src[key]) / size_src
            loss[key] = loss_src[key] / size_src
            print("\t{key}: acc = {acc:.2f}, loss = {loss:.4f}".format(
                key=key, acc=acc[key], loss=loss[key]))
        self.log_scalar(acc, prefix='{}_src_acc'.format(subset))
        self.log_scalar(loss, prefix='{}_src_loss'.format(subset))

        print("Target - {}".format(subset))
        acc, loss = dict(), dict()
        for key in ['di', 'ds', 'ci']:
            acc[key] = float(correct_trg[key]) / size_src
            loss[key] = loss_trg[key] / size_src
            print("\t{key}: acc = {acc:.2f}, loss = {loss:.4f}".format(
                key=key, acc=acc[key], loss=loss[key]))
        self.log_scalar(acc, prefix='{}_trg_acc'.format(subset))
        self.log_scalar(loss, prefix='{}_trg_loss'.format(subset))
