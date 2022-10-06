# DAMF - Moral Foundations Inference with Domain Adapting Ability

## Overview

A supervised method to predict moral foundations from texts.

## Install

run git clone of the repo.

```
git clone https://github.com/fionasguo/DAMF.git
```
setup the package
```
python setup.py install
```

## Dependencies

Will be automatically installed when running setup.py.

All dependencies are in [`requirments.txt`](https://github.com/fionasguo/DAMF/blob/master/requirements.txt)

## Use the code

<!-- dd -->

### Run the example code

####To train (and test):

Put the data (eg. xxxxx.csv) under the folder [`data`](https://github.com/fionasguo/DAMF/tree/master/data)

Put the pretrained language model file (used to initiate DAMF) under the folder [`trained_models`](https://github.com/fionasguo/DAMF/tree/master/trained_models)

Change the config file accordingly, see details below for the config file

Given a single csv data file with all data included, the program will automatically split it into train/val/test sets, run:

```
python3 train_and_test.py -m train_test -c config -i data/all_mf_data.csv -o outputs
```

Given a directory to separate csv files each for a train/val/test set, run:

```
python3 train_and_test.py -m train_test -c config -i data/mf_datasets -o outputs
```

####To test using a trained DAMF model:

Put the data (eg. xxxxx.csv) under the folder [`data`](https://github.com/fionasguo/DAMF/tree/master/data)

Put the trained DAMF model file under the folder [`trained_models`](https://github.com/fionasguo/DAMF/tree/master/trained_models) (e.g. trained_models/ckpt)

Change the config file accordingly

Run:

```
python3 train_and_test.py -m test -c config -i data/all_mf_data.csv -o outputs -t trained_models/ckpt
```

Arguments:

- -m: mode, options: train, test, or train_test
- -c: path to the config file
- -i: input data dir, can be a single csv path including all train, val, test data, or a directory including separate csv files each for a train/val/test set
- -o: output dir, eg. ./outputs
- -t: test_model, an already trained DAMF model file for testing


### Config file
see an example config file [`config`](https://github.com/fionasguo/DAMF/blob/master/config)

All arguments:

- pretrained_dir: str, pretrained LM model for initializing tokenzier and initializing feature encoder, eg. 'trained_models/bert-base-uncased'
- mf_model_dir: (optional) str, previously trained DAMF model path, can be used for testing, eg. 'trained_models/ckpt'
- domain_adapt: bool, whether to use domain adversarial training
- transformation: bool, whether to include a linear transformation module to facilitate domain-invariant feature generation
- reconstruction: bool, whether to include a reconstruction module to keep feature encoder from overly corruption by adversarial training
- semi_supervised: bool, whether to use semi-supervised training where labeled source data and unlabeled target data are both used for training
- train_domain: one str or a list of str, the source domain, eg. 'MFTC', or ['MFTC','congress']
- test_domain: one str or a list of str, the target domain, eg. ['congress']
- n_mf_classes: int, can be 10 (number of moral foundation classes), 5 (don't distinguish between virtues and vices) or 2 (moral vs immoral)
- lr: float, init learning rate
- alpha $\alpha$, beta $\beta$: float, for lr decay function, params to update learning rate: $lr = lr_{init}/((1 +\alpha·p)^\beta)$, where $p = (curr\ epoch − num\ no\ adv)/total\ epoch$
- batch_size: int
- num_epoch: int
- dropout_rate: float
- lambda_trans: float, for the transformation layer - regularization term in loss function
- lambda_rec: float, regularization for reconstruction layer
- gamma $\gamma$: float, rate to update lambda_domain (regularization coef for the domain classifier loss) over epochs, where lambda_domain $= 2/(1 + e^{−\gamma·p})-1$
- num_no_adv: int (>0), after number of epoch with no adversarial training, start to use domain classifier by setting lambda_domain > 0 in the loss
- seed: int
