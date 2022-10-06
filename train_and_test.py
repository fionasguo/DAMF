"""
Domain adversarial network for Moral Foundation inference

Command to run: python3 train_and_test.py -m train_test -c config

All arguments:
    data_dir: str, input data directory, can be set in command args or config file
    output_dir: str, directory to write outputs, including trained model and predictions, can be set in command args or config file
    domain_adapt: bool, whether to use domain adversarial training
    transformation: bool, whether to include a linear transformation module to facilitate domain-invariant feature generation
    reconstruction: bool, whether to include a reconstruction module to keep feature encoder from overly corruption by adversarial training
    semi_supervised: bool, whether to use semi-supervised training where labeled source data and unlabeled target data are both used for training
    train_domain: one str or a list, the source domain, eg. 'MFTC', or ['MFTC','congress']
    test_domain: one str or a list, the target domain, eg. ['congress']
    n_mf_classes: int, can be 10 (number of moral foundation classes) or 2 (moral vs immoral)
    pretrained_dir: str, pretrained LM model for tokenzier and initial feature encoder weights eg. 'models/bert-base-uncased'
    mf_model_dir: str, previou trained model path of this model
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

import os
import time
from datetime import datetime
import logging
import argparse
import transformers

from DAMF import read_data
from DAMF import DomainAdaptTrainer
from DAMF import evaluate
from DAMF import feature_embedding_analysis
from DAMF import read_config, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_command_args(args):
    """Read arguments from command line."""

    parser = argparse.ArgumentParser(
        description='Domain adaptation model for moral foundation inference.')
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        required=True,
                        help='train,test,or train_test')
    parser.add_argument(
        '-c',
        '--config_dir',
        type=str,
        required=True,
        help='configuration file dir that specifies hyperparameters etc')
    parser.add_argument('-i',
                        '--data_dir',
                        type=str,
                        required=True,
                        help='input data directory')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=True,
                        help='output directory')
    parser.add_argument(
        '-t',
        '--test_model',
        type=str,
        required=False,
        help='if testing, it is optional to provide a trained model weight dir'
    )
    command_args = parser.parse_args()

    # mode
    mode = command_args.mode

    ## add dirs into args
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    # data dir
    args['data_dir'] = os.path.join(curr_dir, command_args.data_dir)
    # output_dir
    args['output_dir'] = os.path.join(curr_dir, command_args.output_dir)
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    # config dir
    args['config_dir'] = os.path.join(curr_dir, command_args.config_dir)
    # provided trained model weights for testing, optional
    try:
        args['mf_model_dir'] = os.path.join(curr_dir, command_args.test_model)
    except:
        args['mf_model_dir'] = None

    return mode, args


if __name__ == '__main__':
    # logger
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.DEBUG)

    # args
    args = {}
    mode, args = read_command_args(args)
    print(args)
    args = read_config(os.path.dirname(os.path.realpath(__file__)), args)

    # set seed
    set_seed(args['seed'])

    # load data
    start_time = time.time()
    logging.info('Start processing data...')

    # data should be in a csv file with these columns: 'text','domain' and MF_LABELS
    datasets = read_data(args['data_dir'],
                         args['pretrained_dir'],
                         args['n_mf_classes'],
                         args['train_domain'],
                         args['test_domain'],
                         args['semi_supervised'],
                         seed=args['seed'])

    logging.info(f'Finished processing data. Time: {time.time()-start_time}')

    trainer = None

    if 'train' in mode:
        start_time = time.time()
        logging.info('Start training...')

        # set up trainer
        trainer = DomainAdaptTrainer(datasets, args)

        trainer.train()

        logging.info(
            f"Finished training {args['train_domain']} data. Time: {time.time()-start_time}"
        )

    if 'test' in mode:
        start_time = time.time()
        logging.info('Start processing test data ...')

        logging.info('============ Evaluation on Test Data ============= \n')
        # evaluate with the best model just got from training or with the model given from config
        if trainer is not None:
            eval_model_path = args['output_dir'] + '/best_model.pth'
        elif args.get('mf_model_dir') is not None:
            eval_model_path = args['mf_model_dir']
        else:
            raise ValueError('Please provide a model for evaluation.')
        # check if test data is good
        if 'test' not in datasets or datasets['test'].mf_labels is None:
            raise ValueError('Invalid test dataset.')
        # evaluate
        test_accu = evaluate(datasets['test'],
                             args['batch_size'],
                             model_path=eval_model_path,
                             is_adv=args['domain_adapt'],
                             test=True)
        logging.info('Macro F1 of the %s TEST dataset: %f' %
                     ('target', test_accu))

        # plot feature embedding heapmaps and tsne for domain adapt cases
        if 's_train' in datasets:
            logging.info('performing feature embedding analysis')
            feature_embedding_analysis(datasets['s_train'],
                                       datasets['t_train'],
                                       args['output_dir'],
                                       args['batch_size'],
                                       model_path=eval_model_path)

        logging.info(
            f"Finished evaluating test data {args['test_domain']}. Time: {time.time()-start_time}"
        )
