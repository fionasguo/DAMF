import os
import time
from datetime import datetime
import logging
import argparse
import csv
from ast import literal_eval
import subprocess
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
    parser.add_argument('-c',
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
    parser.add_argument('-s',
                        '--seed_lst',
                        type=str,
                        required=True,
                        help='a list of seeds')

    command_args = parser.parse_args()

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
    # seeds
    args['seed_lst'] = literal_eval(command_args.seed_lst)

    return args


if __name__ == '__main__':
    # logger
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.INFO)

    # args
    args = {}
    args = read_command_args(args)
    args = read_config(os.path.dirname(os.path.realpath(__file__)), args)

    # loop thru seeds
    start_time = time.time()
    test_accus = []

    for seed in args['seed_lst']:
        logging.info('Start evaluating seed %d' % seed)

        args['seed'] = seed
        set_seed(seed)

        # load data
        datasets = read_data(args['data_dir'],
                             args['pretrained_dir'],
                             args['n_mf_classes'],
                             args['train_domain'],
                             args['test_domain'],
                             args['semi_supervised'],
                             args['aflite'],
                             seed=seed,
                             train_frac=0.8)

        trainer = DomainAdaptTrainer(datasets, args)

        trainer.train()

        test_accu,_ = evaluate(datasets['test'],
                               args['batch_size'],
                               model_path=args['output_dir']+'/best_model.pth',
                               is_adv=args['domain_adapt'],
                               test=True
        )
        test_accus.append(test_accu)
        logging.info('Seed = %d. Macro F1 of the target TEST dataset with best model: %f' % (seed, test_accu))

        # clean up best_model file
        subprocess.call(["rm", args['output_dir'] + '/best_model.pth'])

    mean = sum(test_accus) / len(test_accus)
    variance = sum([((x - mean) ** 2) for x in test_accus]) / len(test_accus)
    std_dev = variance ** 0.5
    logging.info('Evaluation over different seeds %s is done. Average F1 = %f; std dev = %f' % (str(args['seed_lst']), mean, std_dev))
    logging.info(f"Time: {time.time()-start_time}")
