# DAMF - Moral Foundations Inference with Domain Adapting Ability

## Overview

A supervised method to predict moral foundations from texts.

## Install

run git clone of the repo.

```
git clone https://github.com/fionasguo/DAMF.git
```

## Dependencies

Install all dependencies in [`requirments.txt`](https://github.com/fionasguo/DAMF/blob/master/requirements.txt)

## Use the code

<!-- dd -->

### Run the code
First, make sure to put training and/or test data under the folder [`data`](https://github.com/fionasguo/DAMF/tree/master/data).

Then, make sure to put vanilla pretrained language model files under the folder [`trained_models`](https://github.com/fionasguo/DAMF/tree/master/trained_models).

Then run:

```
python3 train_and_test.py -m train_test -c config -i mf_data.csv -o tmp
```

Arguments:

- -m: mode, train, test, or train_test 
- -c: path to config file, see details below for the config fiile
- -i: input data dir, eg. mf_data.csv, should be under the folder [`data`](https://github.com/fionasguo/DAMF/tree/master/data)
- -o: output dir, eg. tmp


### Config file
see an example config file [`config`](https://github.com/fionasguo/DAMF/blob/master/config)

All arguments:

- pretrained_dir: str, pretrained LM model for tokenzier and initial feature encoder weights, these should be stored in the folder [`trained_models`](https://github.com/fionasguo/DAMF/tree/master/trained_models), eg. 'bert-base-uncased'
- mf_model_dir: (optional) str, previously trained model path of this model, can be used for testing, these should be stored in the folder [`trained_models`](https://github.com/fionasguo/DAMF/tree/master/trained_models)
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
