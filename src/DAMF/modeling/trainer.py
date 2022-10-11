"""
Trainer class for basic and domain adapt models
"""

import numpy as np
import math
import logging

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from DAMF.data_processing.data_loader import MFData
from .modules import ReconstructionLoss, TransformationLoss
from .model import MFBasic, MFDomainAdapt
from .evaluate import evaluate
from DAMF.utils.feature_analysis import compute_feat
from DAMF.utils.utils import get_gpu_memory_map, count_devices


class DomainAdaptTrainer:

    def __init__(self, datasets, args):
        """
        Args:
            datsets: dict with MFData dataset objects;
                if semi_supervised learning it contains keys: s_train, t_train, s_val, t_val, test
                otherwise it contains keys: train, val, test
            args: dict of trainig args
                num_no_adv: number of epochs trained without adversary
                alpha,beta: params to update learning rate: lr = lr_init/((1 +α·p)^β), where p = (curr_epoch − num_no_adv)/tot_epoch
                lambda_trans: regularization coef for the transformation loss term
                lambda_domain: regularization coef for the domain classifier loss, it's 0 during num_no_adv epochs, but will be updated afterwards
                gamma: rate to update lambda_domain over epochs, lambda_domain = 2/(1 + e^{−γ·p})-1
        """

        self.datasets = datasets
        self.args = args

        # init model
        if self.args['domain_adapt']:
            self.model = MFDomainAdapt(
                self.args['pretrained_dir'], self.args['n_mf_classes'],
                self.args['n_domain_classes'], self.args['dropout_rate'],
                self.args['device'], self.args['transformation'],
                self.args['reconstruction'])
            # set up optimizer & scheduler
            self.create_optimizer_and_scheduler_adversarial_training()

        else:
            self.model = MFBasic(self.args['pretrained_dir'],
                                 self.args['n_mf_classes'],
                                 self.args['dropout_rate'])
            self.create_optimizer_and_scheduler_basic()

        self.model_embedding_dim = self.model.embedding_dim

        # if torch.cuda.device_count() > 1:
        self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.to(self.args['device'])

        for p in self.model.parameters():
            p.requires_grad = True

    def create_optimizer_and_scheduler_basic(self):
        """Create optimizer and scheduler for basic model"""
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args['lr'])

        # scheduler
        def lr_lambda(epoch: int):
            # p = epoch / self.args['n_epoch']
            # decay_factor = 1 / ((1 + self.args['alpha'] * p) ** self.args['beta'])
            decay_factor = 1

            return decay_factor

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def create_optimizer_and_scheduler_adversarial_training(self):
        """
        Create optimizer and scheduler for domain adversarial model

        To update learning rate: lr = lr_init/((1 +α·p)^β), where p = (curr_epoch − num_no_adv)/tot_epoch
        """
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args['lr'])

        # scheduler
        def lr_lambda(epoch: int):
            if epoch >= self.args['num_no_adv']:
                tot_epochs_for_calc = self.args['n_epoch'] - \
                    self.args['num_no_adv']
                epoch_for_calc = epoch - self.args['num_no_adv']
                p = epoch_for_calc / tot_epochs_for_calc
                decay_factor = 1 / \
                    ((1 + self.args['alpha'] * p) ** self.args['beta'])
            else:
                decay_factor = 1

            return decay_factor

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def update_lambda_domain(self, epoch, i_dataloader):
        """
        Update lambda_domain over epochs, lambda_domain = 2/(1 + e^{−γ·p})-1

        lambda_domain is zero for the first num_no_adv epochs.
        Afterwards, update lambda_domain with the increase of n_epochs.
        Starting close to 0, and approaching 1.

        Args:
            epoch: current epoch
            i_dataloader: current position in dataloader iteration
        """
        if epoch >= self.args['num_no_adv']:
            tot_steps_for_calc = (
                self.args['n_epoch'] -
                self.args['num_no_adv']) * self.len_dataloader
            curr_steps = float(i_dataloader +
                               (epoch - self.args['num_no_adv']) *
                               self.len_dataloader)
            p = curr_steps / tot_steps_for_calc

            return 2 / (1 + math.exp(-self.args['gamma'] * p)) - 1
        else:
            return 0.0

    def compute_batch_size(self):
        """
        Compute batch size of source and target data balanced between each domain.

        Because source and target data can contain different numbers of domains, we should balance the batch size in each iteration of training:
        s_train_batch_size ~ user_defined_batch_size
        t_train_batch_size ~ size_of_t_train / size_of_s_train * user_defined_batch_size

        Because maybe we are runing on multiple gpus, need to make sure each gpu gets the same amount.
        """
        # number of devices
        n_devices = count_devices()
        # t_train batch size portioned according to the size of t_train and s_train. dividing, rounding and multiplying by n_devices to make sure each device gets the same amount data
        t_train_batch_size = round(
            self.args['batch_size'] * len(self.datasets['t_train']) /
            len(self.datasets['s_train']) / n_devices) * n_devices
        # in case t_train batch size is too small
        if t_train_batch_size <= n_devices:
            t_train_batch_size = 2 * n_gpus
            s_train_batch_size = math.ceil(
                len(self.datasets['s_train']) /
                (len(self.datasets['t_train']) // t_train_batch_size))
        else:
            s_train_batch_size = self.args['batch_size']

        logging.debug(
            's_train size: %d, s_train batch size: %d, t_train size: %d, t_train batch size: %d'
            % (len(self.datasets['s_train']), s_train_batch_size,
               len(self.datasets['t_train']), t_train_batch_size))

        return s_train_batch_size, t_train_batch_size

    def print_loss(self, loss):
        """Convert loss (if it's a tensor) to a scalar"""
        try:
            return loss.item()
        except:
            return loss

    ########### loss ###########

    def compute_weights(self):
        """
        Compute the weights when calculating mf loss.

        Balance between positive and negative examples, and balance among different classes, with # neg examples / # pos examples
        """
        if self.args['semi_supervised']:
            data_name = 's_train'
        else:
            data_name = 'train'

        n_pos = np.sum(self.datasets[data_name].mf_labels, axis=0).reshape(-1)
        n_pos = np.where(n_pos == 0, 1, n_pos)
        len_labels = len(self.datasets[data_name].mf_labels)
        weights = (len_labels - n_pos) / n_pos
        weights = torch.tensor(weights, dtype=torch.float)
        return weights

    def compute_domain_adapt_loss(self, data, epoch, mf_loss=True):
        """
        Compute loss for domain adapt model.

        Args:
            data: a batch of data
            epoch: current epoch
            mf_loss: bool, target training data doesn't have mf loss
        """
        # before num_no_adv, don't do adversarial training
        is_adv = (epoch >= self.args['num_no_adv'])

        outputs = self.model(data['input_ids'],
                             data['attention_mask'],
                             data['domain_labels'],
                             self.args['lambda_domain'],
                             adv=is_adv)

        loss = 0.0

        # domain classification loss
        loss_domain = 0.0
        if is_adv:
            loss_domain = self.loss_fn_domain(outputs['domain_output'],
                                              data['domain_labels'].squeeze())
            loss += loss_domain
        # mf loss
        loss_mf = 0.0
        if mf_loss:
            loss_mf = self.loss_fn_mf(outputs['class_output'],
                                      data['mf_labels'])
            loss += loss_mf
        # rec loss
        loss_rec = 0.0
        if self.args['reconstruction'] and is_adv:
            # when training adversarially, compute loss between original and current feature embeddings
            loss_rec = self.args['lambda_rec'] * self.loss_fn_rec(
                data['feat_embed'], outputs['rec_embed'])
            loss += loss_rec
        # trans loss - don't need this for target data
        loss_trans = 0.0
        if self.args['transformation'] and is_adv and mf_loss:
            loss_trans = self.args['lambda_trans'] * \
                self.loss_fn_trans(self.model.module.trans_module.l.weight)
                ## TODO: modify this
            # loss += loss_trans

        return loss, loss_mf, loss_domain, loss_rec, loss_trans

    ########### train ###########

    def train(self):
        if self.args['semi_supervised']:
            accu = self.train_semisupervised()
        else:
            accu = self.train_basic()

        return accu

    def train_basic(self):
        # set up dataloader
        # first set worker init fn
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # init dataloader
        self.train_dataloader = DataLoader(dataset=self.datasets['train'],
                                           batch_size=self.args['batch_size'],
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=4,
                                           worker_init_fn=worker_init_fn)

        # loss fn
        self.loss_fn_mf = torch.nn.BCEWithLogitsLoss().to(self.args['device'])
        if self.args['domain_adapt']:
            self.loss_fn_domain = torch.nn.CrossEntropyLoss().to(
                self.args['device'])
            self.loss_fn_trans = TransformationLoss(
                self.model_embedding_dim, self.args['device']
            ).to(self.args['device']) if self.args['transformation'] else None
            self.loss_fn_rec = ReconstructionLoss().to(
                self.args['device']) if self.args['reconstruction'] else None

        self.len_dataloader = len(self.train_dataloader)

        best_accu = 0.0
        best_epoch = 0

        for epoch in range(self.args['n_epoch']):

            self.model.train()

            is_adv = False
            if self.args['domain_adapt']:
                is_adv = (epoch >= self.args['num_no_adv'])

            data_iter = iter(self.train_dataloader)

            for i in range(self.len_dataloader):
                data, _ = data_iter.next()
                for k, v in data.items():
                    data[k] = data[k].to(self.args['device'])

                self.model.zero_grad()

                loss_domain, loss_rec, loss_trans = 0.0, 0.0, 0.0
                if not self.args['domain_adapt']:
                    outputs = self.model(data['input_ids'],
                                         data['attention_mask'])
                    loss_mf = self.loss_fn_mf(outputs['class_output'],
                                              data['mf_labels'])
                    loss = loss_mf
                else:
                    loss, loss_mf, loss_domain, loss_rec, loss_trans = self.compute_domain_adapt_loss(
                        data, epoch)

                loss.backward()
                self.optimizer.step()

                # print loss values
                logging.debug(
                    '\r epoch: %d, [iter: %d / all %d], total_loss: %.3f, loss_mf: %.3f, loss_domain: %.3f, loss_rec: %.3f, loss_trans: %.3f'
                    %
                    (epoch, i + 1, self.len_dataloader, self.print_loss(loss),
                     self.print_loss(loss_mf), self.print_loss(loss_domain),
                     self.print_loss(loss_rec), self.print_loss(loss_trans)))

                # # save temp model
                # torch.save(self.model.module,
                #            self.args['output_dir'] + '/model_in_training.pth')

            # test on validation set
            accu = evaluate(self.datasets['val'],
                            self.args['batch_size'],
                            model=self.model.module,
                            is_adv=is_adv)
            logging.info(f'\nepoch: {epoch}')
            logging.info('Macro F1 of the VAL dataset: %.3f' % accu)

            if accu > best_accu:
                best_accu = accu
                best_epoch = epoch
                torch.save(self.model.module,
                           self.args['output_dir'] + '/best_model.pth')

            if self.args['reconstruction'] and epoch == (
                    self.args['num_no_adv'] - 1):
                # compute feature embeddings and save in the dataset object
                compute_feat(self.model.module, self.datasets['train'],
                             self.args['device'], self.args['batch_size'])

            # update lr scheduler
            self.scheduler.step()

            # log cpu/gpu usage
            device_info = get_gpu_memory_map()
            logging.info('GPU/CPU usage (MB): %s' % device_info)

        logging.info('============ Training Summary ============= \n')
        logging.info(
            f'Best Macro F1 of the VAL dataset: {best_accu} at epoch {best_epoch}'
        )
        logging.info('Corresponding model was save in ' +
                     self.args['output_dir'] + '/best_model.pth')

        return best_accu

    def train_semisupervised(self):
        """if training with semi-supervised method, we are surely doing domain adversarial training"""
        # set up dataloader
        # first calculate batch size for source and target domains
        s_train_batch_size, t_train_batch_size = self.compute_batch_size()

        # set worker init fn

        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # init dataloaders
        self.dataloaders = {}
        self.dataloaders['s_train'] = DataLoader(
            dataset=self.datasets['s_train'],
            batch_size=s_train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            worker_init_fn=worker_init_fn)
        self.dataloaders['t_train'] = DataLoader(
            dataset=self.datasets['t_train'],
            batch_size=t_train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            worker_init_fn=worker_init_fn)
        self.len_dataloader = min(len(self.dataloaders['s_train']),
                                  len(self.dataloaders['t_train']))

        # loss fn
        self.loss_fn_mf = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.compute_weights()).to(self.args['device'])
        self.loss_fn_domain = torch.nn.CrossEntropyLoss().to(
            self.args['device'])
        self.loss_fn_trans = TransformationLoss(
            self.model_embedding_dim, self.args['device']).to(
                self.args['device']) if self.args['transformation'] else None
        self.loss_fn_rec = ReconstructionLoss().to(
            self.args['device']) if self.args['reconstruction'] else None

        logging.debug('lambda_trans = %f, lambda_rec = %f' %
                      (self.args['lambda_trans'], self.args['lambda_rec']))

        # training

        best_accu_s = 0.0
        best_accu_t = 0.0
        best_epoch = 0

        for epoch in range(self.args['n_epoch']):

            self.model.train()

            is_adv = (epoch >= self.args['num_no_adv'])

            data_source_iter = iter(self.dataloaders['s_train'])
            if is_adv:
                data_target_iter = iter(self.dataloaders['t_train'])

            for i in range(self.len_dataloader):
                # update lambda_domain
                self.args['lambda_domain'] = self.update_lambda_domain(
                    epoch, i)
                logging.debug(
                    '\r epoch: %d, [iter: %d / all %d], lambda_domain = %f' %
                    (epoch, i + 1, self.len_dataloader,
                     self.args['lambda_domain']))

                # get data
                data_source, _ = data_source_iter.next()
                for k, v in data_source.items():
                    data_source[k] = data_source[k].to(self.args['device'])

                if is_adv:
                    data_target, _ = data_target_iter.next()
                    for k, v in data_target.items():
                        data_target[k] = data_target[k].to(self.args['device'])

                self.model.zero_grad()

                # training model using source data
                s_loss, s_loss_mf, s_loss_domain, s_loss_rec, s_loss_trans = self.compute_domain_adapt_loss(
                    data_source, epoch, mf_loss=True)
                loss = s_loss.clone()
                # training model using target data
                t_loss, t_loss_mf, t_loss_domain, t_loss_rec, t_loss_trans = 0.0, 0.0, 0.0, 0.0, 0.0
                if is_adv:
                    t_loss, t_loss_mf, t_loss_domain, t_loss_rec, t_loss_trans = self.compute_domain_adapt_loss(
                        data_target, epoch, mf_loss=False)
                    # balance loss between source and target
                    loss = loss + (s_train_batch_size /
                                   t_train_batch_size) * t_loss + s_loss_trans

                loss.backward()
                self.optimizer.step()

                # print loss values
                logging.debug(
                    '\r epoch: %d, [iter: %d / all %d], total_loss: %.3f, total_s_loss: %.3f, total_t_loss: %.3f\ns_loss_mf: %.3f, s_loss_domain: %.3f, s_loss_rec: %.3f, s_loss_trans: %.3f\nt_loss_domain: %.3f, t_loss_rec: %.3f, t_loss_trans: %.3f'
                    % (epoch, i + 1, self.len_dataloader,
                       self.print_loss(loss), self.print_loss(s_loss),
                       (s_train_batch_size / t_train_batch_size) *
                       self.print_loss(t_loss), self.print_loss(s_loss_mf),
                       self.print_loss(s_loss_domain),
                       self.print_loss(s_loss_rec),
                       self.print_loss(s_loss_trans),
                       self.print_loss(t_loss_domain),
                       self.print_loss(t_loss_rec),
                       self.print_loss(t_loss_trans)))

                del loss

                # # save temp model
                # torch.save(self.model.module,
                #            self.args['output_dir'] + '/model_in_training.pth')

            # test on source validation set
            logging.info(f'\nepoch: {epoch}')
            accu_s = evaluate(self.datasets['s_val'],
                              self.args['batch_size'],
                              model=self.model.module,
                              is_adv=is_adv)
            logging.info('Macro F1 of the %s dataset: %f' % ('source', accu_s))
            # test on target val set if its label exists
            if 't_val' in self.datasets and self.datasets[
                    't_val'].mf_labels is not None:
                accu_t = evaluate(self.datasets['t_val'],
                                  self.args['batch_size'],
                                  model=self.model.module,
                                  is_adv=is_adv)
                logging.info('Macro F1 of the %s dataset: %f\n' %
                             ('target', accu_t))
                if accu_t > best_accu_t and epoch >= self.args['num_no_adv']:
                    best_accu_s = accu_s
                    best_accu_t = accu_t
                    best_epoch = epoch
                    torch.save(self.model.module,
                               self.args['output_dir'] + '/best_model.pth')
            else:
                if accu_s > best_accu_s and epoch >= self.args['num_no_adv']:
                    best_accu_s = accu_s
                    best_epoch = epoch
                    torch.save(self.model.module,
                               self.args['output_dir'] + '/best_model.pth')

            # if computing rec loss, at the end of non adversarial training, record the model as the basic mf predicting model without domain adapt, the feature embeddings calculated from this mode will be the original embed
            if self.args['reconstruction'] and epoch == (
                    self.args['num_no_adv'] - 1):
                # compute feature embeddings and save in the dataset object
                compute_feat(self.model.module, self.datasets['s_train'],
                             self.args['device'], self.args['batch_size'])
                compute_feat(self.model.module, self.datasets['t_train'],
                             self.args['device'], self.args['batch_size'])

            # update lr scheduler
            self.scheduler.step()

            # log cpu/gpu usage
            device_info = get_gpu_memory_map()
            logging.info('GPU/CPU usage (MB): %s' % device_info)

        logging.info('============ Training Summary ============= \n')
        logging.info('Best Macro F1 of the %s VAL dataset: %f' %
                     ('source', best_accu_s))
        logging.info('Best Macro F1 of the %s VAL dataset: %f' %
                     ('target', best_accu_t))
        logging.info(f'Best Macro F1 at epoch {best_epoch}')
        logging.info('Corresponding model was save in ' +
                     self.args['output_dir'] + '/best_model.pth')

        return best_accu_t
