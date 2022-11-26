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
    parser.add_argument('--lambda_trans_lst',
                        type=str,
                        required=True,
                        help='a list')
    parser.add_argument('--lambda_rec_lst',
                        type=str,
                        required=True,
                        help='a list')
    parser.add_argument('--gamma_lst',
                        type=str,
                        required=True,
                        help='a list')

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
    # hp
    args['lambda_trans_lst'] = literal_eval(command_args.lambda_trans_lst)
    args['lambda_rec_lst'] = literal_eval(command_args.lambda_rec_lst)
    args['gamma_lst'] = literal_eval(command_args.gamma_lst)

    return args


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
    args = read_command_args(args)
    args = read_config(os.path.dirname(os.path.realpath(__file__)), args)

    # loop thru seeds
    start_time = time.time()
    best_accu = 0.0
    for lambda_trans in args['lambda_trans_lst']:
        for lambda_rec in args['lambda_rec_lst']:
            for gamma in args['gamma_lst']:
                logging.info(f"\nStart HP search: lambda_trans={lambda_trans}, lambda_rec={lambda_rec}, gamma={args['gamma']}")

                args['lambda_rec'] = lambda_rec
                args['lambda_trans'] = lambda_trans
                args['gamma'] = gamma

                datasets = load_data(args['data_dir'],
                                     args['pretrained_dir'],
                                     args['n_mf_classes'],
                                     args['train_domain'],
                                     args['test_domain'],
                                     args['semi_supervised'],
                                     seed=args['seed'],
                                     train_frac=0.8)

                trainer = DomainAdaptTrainer(datasets, args)
                accu = trainer.train()

                test_accu,_ = evaluate(datasets['test'],
                                    args['batch_size'],
                                    model_path=args['output_path']+'/best_model.pth',
                                    domain_adapt=args['domain_adapt'],
                                    test=True
                )
                logging.info(f"\nHP search result: lambda_trans={lambda_trans}, lambda_rec={lambda_rec}, gamma={args['gamma']}, num_no_adv={args['num_no_adv']}, val accu={accu}, test accu={test_accu}")
                if accu > best_accu:
                    best_accu = accu
                    best_lambda_rec = lambda_rec
                    best_lambda_trans = lambda_trans
                    best_gamma = gamma

                    subprocess.call(['cp',args['output_path']+'/best_model.pth',args['output_path']+'/hp_search_best_model.pth'])

                    # clean up best_model file
                    subprocess.call(["rm", args['output_path'] + '/best_model.pth'])

    logging.info(f"\nHyperparameter search finished. Best model has lambda_trans={best_lambda_trans}, lambda_rec={best_lambda_rec}, gamma={args['gamma']}, num_no_adv={args['num_no_adv']}. Macro F1={best_accu}\n")

    logging.info('============ Evaluation on Test Data ============= \n')
    test_accu = evaluate(datasets['test'],
                        args['batch_size'],
                        model_path=args['output_path']+'/hp_search_best_model.pth',
                        domain_adapt=args['domain_adapt'],
                        test=True
    )
    logging.info('Macro F1 of the %s TEST dataset with given (best) model: %f' % ('target', test_accu))
    logging.info(f"Finished evaluating test data {args['test_domain']}. Time: {time.time()-start_time}")
