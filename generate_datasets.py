"""
generate train/dev/test datasets from a csv file
the csv needs to have these columns: 'text','domain' and mf_labels
if using a domain adapt model, then the generated datasets are train_source, train_target, etc
"""

import pandas as pd
import numpy as np
from ast import literal_eval

from preprocessing import preprocess_tweet
from AFLite import run_aflite
from data_loader import MFData

from transformers import AutoTokenizer

pd.options.mode.chained_assignment = None

def gen_in_domain_data(df,train_domain,test_domain,train_frac=0.8,seed=3):
    """
    train_domain should be a superset of test_domain
    eg train_domain=['MFTC','congress'],test_domain=['MFTC'] or test_domain=['MFTC','congress']
    """
    df = df[df.domain.isin(train_domain)]
    # assign domain labels
    df['domain_idx'] = df.domain.apply(lambda x: train_domain.index(x))

    # training, val, test
    df = df.sample(frac=1,random_state=seed) # shuffle first

    test = df[df.domain.isin(test_domain)]
    test = test.sample(frac=1-train_frac,random_state=seed)
    train = df.drop(test.index)


    return train,test

def gen_out_domain_data(df,train_domain,test_domain,train_frac=0.8,seed=3):
    """
    test domain not included in train domain
    """
    df = df.sample(frac=1,random_state=seed) # shuffle

    # training data
    train = df[df.domain.isin(train_domain)]
    # assign domain labels
    train['domain_idx'] = train.domain.apply(lambda x: train_domain.index(x))

    # test data
    test = df[df.domain.isin(test_domain)]
    test['domain_idx'] = test.domain.apply(lambda x: test_domain.index(x)+len(train_domain))

    return train,test

def gen_semi_supervised(train,val,test,train_frac=0.8,seed=3,balance=True):
    """
    further separate train/dev/test data by source and target
    balance: bool, whether to balance the size of data from each domain
    """
    # test data are the target data, save a small portion as the true test data,
    # make the majority of target data the unlabeled training data
    unlabeled_training_data = test.sample(frac=train_frac,random_state=seed)
    # for the unlabeled training data, split into train and val
    target_train = unlabeled_training_data.sample(frac=train_frac,random_state=seed)
    target_val = unlabeled_training_data.drop(target_train.index)
    # save a small portion as the true test data
    true_test_data = test.drop(unlabeled_training_data.index)

    if balance:
        # in the training/val data, balance each domain in the source data and target data
        # have to sample every domain to the size of the smallest domain
        train_domain_size = min(
            train.groupby('domain_idx')['text'].count().min(),
            target_train.groupby('domain_idx')['text'].count().min()
        )
        train = train.groupby('domain_idx').sample(n=train_domain_size,random_state=seed)
        target_train = target_train.groupby('domain_idx').sample(n=train_domain_size,random_state=seed)

        val_domain_size = min(
            val.groupby('domain_idx')['text'].count().min(),
            target_val.groupby('domain_idx')['text'].count().min()
        )
        val = val.groupby('domain_idx').sample(n=val_domain_size,random_state=seed)
        target_val = target_val.groupby('domain_idx').sample(n=val_domain_size,random_state=seed)

    return {'s_train':train,'t_train':target_train,'s_val':val,'t_val':target_val,'test':true_test_data}

def gen_datasets(df, n_mf_classes, train_domain=None, test_domain=None, aflite=False, semi_supervised=False, balance=False, train_frac=0.8, seed=3):
    """
    main function to generate train/dev/test datasets

    train_domain/test_domain: the names of these domains,
        should be in the 'domain' column in the source csv
        should be a list of str
    semi_supervised: if true, generate train_source, train_target, etc
    train_frac: the fraction of data to be used as training data

    return: dict of dicts
    {
     'train':
             {
              'text':[...],
              'labels':[...]
             },
     'val':{...},
     'test':{...}
    }
    """

    if n_mf_classes == 2:
        mf_labels = ['moral','immoral']
    elif n_mf_classes == 10:
        mf_labels = ['care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity', 'degradation']
    elif n_mf_classes == 5:
        mf_labels = ['care', 'fairness', 'loyalty', 'authority', 'purity']
    else:
        raise ValueError('n_mf_classes can be 2 (moral vs immoral) or 10 (all MF classes)')

    if set(test_domain).issubset(set(train_domain)):
        train,test = gen_in_domain_data(df,train_domain,test_domain,train_frac,seed)
    else:
        train,test = gen_out_domain_data(df,train_domain,test_domain,train_frac,seed)

    if aflite:
        train = run_aflite(train,mf_labels)

    val = train.sample(frac=(1-train_frac),random_state=seed)
    train = train.drop(val.index)

    dataset_dict = {'train':train,'val':val,'test':test}

    if semi_supervised:
        dataset_dict = gen_semi_supervised(train,val,test,train_frac=train_frac,seed=seed,balance=balance)

    # take the text and labels
    datasets = {}
    for k,v in dataset_dict.items():
        datasets[k] = {}

        datasets[k]['encodings'] = v.encodings.tolist()
        datasets[k]['mf_labels'] = v[mf_labels].values
        datasets[k]['domain_labels'] = v['domain_idx'].values

    return datasets

def load_data(data_dir, max_seq_len, tokenizer_path, n_mf_classes, aflite, semi_supervised, train_domain=None, test_domain=None,train_frac=0.8,seed=3,balance=True):
    """
    data_dir: should be a csv with these columns: 'text','domain' and MF_LABELS
    max_seq_length: max number of tokens from each input text; used by the tokenizer
    tokenizer_path: where to load pretrained tokenizer
    n_mf_classes: number of total moral classes, could be all moral foundations or just moral vs immoral
    semi_supervised: bool, whether to use semi-supervised training where labeled source data and unlabeled target data are both used for training
    train_domain/test_domain: the names of these domains, should be in the 'domain' column in the source csv
    train_frac: the fraction of data to be used as training data
    balance: if semi_supervised=True, whether to balance data size from different domains

    return: dict of train,val and/or test MFData objects
    """
    df = pd.read_csv(data_dir)

    df.text = df.text.apply(preprocess_tweet)
    df = df[df.text!='']
    df = df.reset_index()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    df['encodings'] = df['text'].apply(tokenizer, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors='pt')

    datasets = {}

    # call function to generate a dict of train,dev,test dicts
    # each with keys - 'text' and 'labels', values are lists
    # also look at the number of mf and domain classes
    datasets = gen_datasets(df,n_mf_classes,train_domain,test_domain,aflite,semi_supervised,balance,train_frac,seed)

    # text preprocessing; turn into MFData object
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)

    for i in datasets.keys():
        # n_domain_classes = len(set(i[-1] for i in datasets[i]['labels']))
        # encodings = tokenizer(datasets[i]['text'], truncation=True, max_length=max_seq_len, padding="max_length")
        datasets[i] = MFData(datasets[i]['encodings'], datasets[i]['mf_labels'], datasets[i]['domain_labels']) #, n_domain_classes)

    return datasets
