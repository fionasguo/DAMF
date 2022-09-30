"""
Domain adversarial network for Moral Foundation inference

All arguments:
    data_dir: str, input data directory, can be set in command args or config file
    output_path: str, directory to write outputs, including trained model and predictions, can be set in command args or config file
    domain_adapt: bool, whether to use domain adversarial training
    transformation: bool, whether to include a linear transformation module to facilitate domain-invariant feature generation
    reconstruction: bool, whether to include a reconstruction module to keep feature encoder from overly corruption by adversarial training
    semi_supervised: bool, whether to use semi-supervised training where labeled source data and unlabeled target data are both used for training
    train_domain: one str or a list, the source domain, eg. 'MFTC', or ['MFTC','congress']
    test_domain: one str or a list, the target domain, eg. ['congress']
    n_mf_classes: int, can be 10 (number of moral foundation classes) or 2 (moral vs immoral)
    pretrained_path: str, pretrained LM model for tokenzier and initial feature encoder weights eg. 'models/bert-base-uncased'
    mf_model_path: str, previou trained model path of this model
    lr: learning rate
    alpha, beta: float, for lr decay function, params to update learning rate: lr = lr_init/((1 +α·p)^β), where p = (curr_epoch − num_no_adv)/tot_epoch
    max_seq_len: int, max sequence length for tokenizer
    batch_size: int
    num_epoch: int
    dropout_rate: float
    lambda_trans: float, for the transformation layer - regularization term in loss function
    lambda_rec: float, regularization for reconstruction layer
    num_no_adv: int (>0), after number of epoch with no adversarial training, start to use domain classifier by setting lambda_domain > 0 in the loss
    gamma: float, rate to update lambda_domain (regularization coef for the domain classifier loss) over epochs, lambda_domain = 2/(1 + e^{−γ·p})-1
    seed: int

"""

import random
import os
import sys
import time
import argparse
from ast import literal_eval
import numpy as np
import subprocess
from datetime import datetime
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

import transformers

from generate_datasets import load_data
from trainer import DomainAdaptTrainer
from evaluate import evaluate
from utils import read_args, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    ## logger
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.INFO)

    ## args
    mode, args = read_args()

    ## set seed
    set_seed(args['seed'])

    ## load data
    start_time = time.time()
    logging.info('Start processing data...')

    # data should be in a csv file with these columns: 'text','domain' and MF_LABELS
    datasets = load_data(args['data_dir'], args['pretrained_path'],
                         args['n_mf_classes'], args['train_domain'],
                         args['test_domain'], args['semi_supervised'],
                         args['seed'])

    logging.info(f'Finished processing data. Time: {time.time()-start_time}')

    ## train
    trainer = None

    if mode == 'hp_search':
        start_time = time.time()
        logging.info('Start hyperparameter searching...')

        best_accu = 0.0
        for lambda_rec in [1, 0.1]:
            for lambda_trans in [10, 1, 0.1, 0.01]:
                args['lambda_rec'] = lambda_rec
                args['lambda_trans'] = lambda_trans
                # args['gamma'] = gamma

                datasets = load_data(args['data_dir'], args['pretrained_path'],
                                     args['n_mf_classes'],
                                     args['train_domain'], args['test_domain'],
                                     args['semi_supervised'], args['seed'])

                trainer = DomainAdaptTrainer(datasets, args)
                accu = trainer.train()

                logging.info(
                    f"\nHyperparameter search: lambda_trans={lambda_trans}, lambda_rec={lambda_rec}, gamma={args['gamma']}, num_no_adv={args['num_no_adv']}, val accu={accu}"
                )
                test_accu = evaluate(datasets['test'],
                                     args['batch_size'],
                                     model_path=args['output_path'] +
                                     '/best_model.pth',
                                     domain_adapt=args['domain_adapt'],
                                     test=True)
                logging.info('test accu: %f' % test_accu)
                if accu > best_accu:
                    best_accu = accu
                    best_lambda_rec = lambda_rec
                    best_lambda_trans = lambda_trans

                    subprocess.call([
                        'cp', args['output_path'] + '/best_model.pth',
                        args['output_path'] + '/hp_search_best_model.pth'
                    ])

        logging.info(
            f"\nHyperparameter search finished. Best model has lambda_trans={best_lambda_trans}, lambda_rec={best_lambda_rec}, gamma={args['gamma']}, num_no_adv={args['num_no_adv']}. Macro F1={best_accu}\n"
        )
        logging.info('============ Evaluation on Test Data ============= \n')
        test_accu = evaluate(datasets['test'],
                             args['batch_size'],
                             model_path=args['output_path'] +
                             '/hp_search_best_model.pth',
                             domain_adapt=args['domain_adapt'],
                             test=True)
        logging.info(
            'Macro F1 of the %s TEST dataset with given (best) model: %f' %
            ('target', test_accu))
        logging.info(
            f"Finished evaluating test data {args['test_domain']}. Time: {time.time()-start_time}"
        )

    elif mode == 'evaluate_seeds':
        start_time = time.time()
        seeds = [3, 2, 10, 910,
                 5408]  # [1,2,10,11,13,15] #[7890,2891,89154,456,189]
        if 'seed_lst' in args: seeds = args['seed_lst']
        logging.info('Start evaluating over seed: %s' % str(seeds))

        test_accus = []
        for s in seeds:  # 57,910,4897,23923 # [19,37,932,1790,854]:# [3,29,2022,3901,5408]:
            logging.info("seed = %d" % s)
            args['seed'] = s
            set_seed(s)

            datasets = load_data(args['data_dir'], args['pretrained_path'],
                                 args['n_mf_classes'], args['train_domain'],
                                 args['test_domain'], args['semi_supervised'],
                                 args['seed'])

            trainer = DomainAdaptTrainer(datasets, args)
            accu = trainer.train()

            test_accu = evaluate(datasets['test'],
                                 args['batch_size'],
                                 model_path=args['output_path'] +
                                 '/best_model.pth',
                                 domain_adapt=args['domain_adapt'],
                                 test=True)
            test_accus.append(test_accu)
            logging.info(
                'Seed = %f. Macro F1 of the %s TEST dataset with best model: %f'
                % (s, 'target', test_accu))

        mean = sum(test_accus) / len(test_accus)
        variance = sum([((x - mean)**2) for x in test_accus]) / len(test_accus)
        std_dev = variance**0.5
        logging.info(
            'Evaluation over different seeds %s is done. Average F1 = %f; std dev = %f'
            % (str(seeds), mean, std_dev))
        logging.info(f"Time: {time.time()-start_time}")

    else:
        if mode != 'test':
            start_time = time.time()
            logging.info('Start training...')

            # set up trainer
            trainer = DomainAdaptTrainer(datasets, args)

            trainer.train()

            logging.info(
                f"Finished training {args['train_domain']} data. Time: {time.time()-start_time}"
            )

        if mode != 'train':
            start_time = time.time()
            logging.info('Start processing test data ...')

            logging.info(
                '============ Evaluation on Test Data ============= \n')
            # evaluate with the best model
            try:
                test_accu = evaluate(datasets['test'],
                                     args['batch_size'],
                                     model_path=args['output_path'] +
                                     '/best_model.pth',
                                     domain_adapt=args['domain_adapt'],
                                     test=True)
                logging.info(
                    'Macro F1 of the %s TEST dataset with model at last epoch: %f'
                    % ('target', test_accu))
                # plot feature embedding heapmaps and tsne for domain adapt cases
                if 's_val' in datasets:
                    logging.info('performing feature embedding analysis')
                    feature_embedding_analysis(datasets['s_val'],
                                               datasets['t_val'],
                                               args['output_path'],
                                               args['batch_size'],
                                               model_path=args['output_path'] +
                                               '/best_model.pth')
            except:
                pass

            if args.get('mf_model_path') != None:
                # evaluate with the given model path
                test_accu = evaluate(datasets['test'],
                                     args['batch_size'],
                                     model_path=args['mf_model_path'],
                                     domain_adapt=args['domain_adapt'],
                                     test=True)
                logging.info(
                    'Macro F1 of the %s TEST dataset with given (best) model: %f'
                    % ('target', test_accu))
                # plot feature embedding heapmaps and tsne for domain adapt cases
                if 's_val' in datasets:
                    logging.info('performing feature embedding analysis')
                    feature_embedding_analysis(
                        datasets['s_val'],
                        datasets['t_val'],
                        args['output_path'],
                        args['batch_size'],
                        model_path=args['mf_model_path'])

            logging.info(
                f"Finished evaluating test data {args['test_domain']}. Time: {time.time()-start_time}"
            )
