"""
Generate train/dev/test datasets from a csv file.

The data csv needs to have these columns: 'text','domain' and all columns in desired mf_label_names.
"""

import pandas as pd
from typing import List, Tuple, Dict

from transformers import AutoTokenizer

from src.data_processing.preprocessing import preprocess_tweet
from src.data_processing.data_loader import MFData

pd.options.mode.chained_assignment = None


def gen_in_domain_data(
        df: pd.DataFrame,
        train_domain: List[str],
        test_domain: List[str],
        train_frac: float = 0.8,
        seed: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate in-domain training/test datasets.

    Domain index will be assigned to different domains.

    Train_domain should be a superset of test_domain. (eg train_domain=['MFTC','congress'], test_domain=['MFTC'] or test_domain=['MFTC','congress'])

    Args:
        df: pandas dataframe with these columns: 'text','domain' and mf_labels (e.g. care, harm).
        train_domain: the name of the domains for training
        test_domain: the name of the domains for testing
        train_frac: the fraction of number of data points used for training
        seed: random seed

    Returns:
        train: a pandas df, training set
        val: a pandas df, validation set
        test: a pandas df, test set
    """
    df = df[df.domain.isin(train_domain)]
    # assign domain index
    df['domain_idx'] = df.domain.apply(lambda x: train_domain.index(x))
    # shuffle
    df = df.sample(frac=1, random_state=seed)
    # get train, val and test set
    test = df[df.domain.isin(test_domain)]
    test = test.sample(frac=(1 - train_frac), random_state=seed)
    train = df.drop(test.index)
    val = train.sample(frac=(1 - train_frac), random_state=seed)
    train = train.drop(val.index)

    return train, val, test


def gen_out_domain_data(
        df: pd.DataFrame,
        train_domain: List[str],
        test_domain: List[str],
        train_frac: float = 0.8,
        seed: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate out-of-domain training/test datasets.

    Domain index will be assigned to different domains.

    Test domain should not be included in train domain. (eg train_domain=['MFTC','congress'], test_domain=['covid'])

    Args:
        df: pandas dataframe with these columns: 'text','domain' and mf_labels (e.g. care, harm).
        train_domain: the name of the domains for training
        test_domain: the name of the domains for testing
        train_frac: the fraction of number of data points used for training
        seed: random seed

    Returns:
        train: a pandas df, training set
        va;: a pandas df, validation set
        test: a pandas df, test set

    """
    # shuffle
    df = df.sample(frac=1, random_state=seed)
    # training data
    train = df[df.domain.isin(train_domain)]
    # assign domain index in train test
    train['domain_idx'] = train.domain.apply(lambda x: train_domain.index(x))
    # val data
    val = train.sample(frac=(1 - train_frac), random_state=seed)
    train = train.drop(val.index)
    # test data
    test = df[df.domain.isin(test_domain)]
    test['domain_idx'] = test.domain.apply(
        lambda x: test_domain.index(x) + len(train_domain))

    return train, val, test


def gen_semi_supervised(train: pd.DataFrame,
                        val: pd.DataFrame,
                        test: pd.DataFrame,
                        train_frac: float = 0.8,
                        seed: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Separate train/dev/test data by source and target domains.

    Args:
        train: training set
        val: validation set
        test: test set
        train_frac: the fraction of number of data points used for training
        seed: random seed

    Returns:
        a dictionary with source training set (s_train), source validation set (s_val) which have text, mf labels and domain labels; target training set (t_train), target validation set (t_val) which have text and only domain labels, and true test set (test) which has text, mf and domain labels.
    """
    # test data are the target data, save a small portion as the true test data,
    # and then make the majority of target data the unlabeled training data
    unlabeled_training_data = test.sample(frac=train_frac, random_state=seed)
    # for the unlabeled training data, split into train and val
    target_train = unlabeled_training_data.sample(frac=train_frac,
                                                  random_state=seed)
    target_val = unlabeled_training_data.drop(target_train.index)
    # save the small portion as the true test data
    true_test_data = test.drop(unlabeled_training_data.index)

    # in the training/val data, balance the amount of data in each domain in the source data and target data, sample every domain to the size of the smallest domain
    train_domain_size = min(
        train.groupby('domain_idx')['text'].count().min(),
        target_train.groupby('domain_idx')['text'].count().min())

    train = train.groupby('domain_idx').sample(n=train_domain_size,
                                               random_state=seed)
    target_train = target_train.groupby('domain_idx').sample(
        n=train_domain_size, random_state=seed)

    val_domain_size = min(
        val.groupby('domain_idx')['text'].count().min(),
        target_val.groupby('domain_idx')['text'].count().min())

    val = val.groupby('domain_idx').sample(n=val_domain_size,
                                           random_state=seed)
    target_val = target_val.groupby('domain_idx').sample(n=val_domain_size,
                                                         random_state=seed)

    return {
        's_train': train,
        't_train': target_train,
        's_val': val,
        't_val': target_val,
        'test': true_test_data
    }


def create_MFData(df: pd.DataFrame,
                  tokenizer: AutoTokenizer,
                  mf_label_names: List[str],
                  max_seq_len: int = 50) -> MFData:
    """
    Create MFData instance from a df.
    """
    encodings = df['text'].apply(tokenizer,
                                 truncation=True,
                                 max_length=max_seq_len,
                                 padding="max_length").tolist()

    mf_labels = df[mf_label_names].values

    domain_labels = df['domain_idx'].values

    mf_data = MFData(encodings, mf_labels, domain_labels)

    return mf_data


def load_data(data_dir: str,
              tokenizer_path: str,
              n_mf_classes: int,
              train_domain: List[str],
              test_domain: List[str],
              semi_supervised: bool,
              max_seq_len: int = 50,
              train_frac: float = 0.8,
              seed: int = 3) -> Dict[str, MFData]:
    """
    Generate train/dev/test datasets.

    Args:
        data_dir: should be a csv with these columns: 'text','domain' and mf_label_names
        tokenizer_path: where to load pretrained tokenizer
        n_mf_classes: number of moral foundation classes in the data
        train_domain: the name of the domains for training, should be in the 'domain' column in df
        test_domain: the name of the domains for testing
        semi_supervised: if true, separate train/dev/test data by source and target domains
        max_seq_length: max number of tokens from each input text used by the tokenizer
        train_frac: the fraction of number of data points used for training
        seed: random seed

    Returns:
        dictionary of train, val, test MFData objects
    """
    # decide the MF labels
    if n_mf_classes == 2:
        mf_label_names = ['moral', 'immoral']
    elif n_mf_classes == 10:
        mf_label_names = [
            'care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal',
            'authority', 'subversion', 'purity', 'degradation'
        ]
    elif n_mf_classes == 5:
        mf_label_names = ['care', 'fairness', 'loyalty', 'authority', 'purity']
    else:
        raise ValueError('n_mf_classes not valid. Choose from 2, 5 or 10.')

    # read data
    df = pd.read_csv(data_dir)
    # preprocess
    df.text = df.text.apply(preprocess_tweet)
    df = df[df.text != '']
    df = df.reset_index()

    # generate train/val/test datasets
    if set(test_domain).issubset(set(train_domain)):
        # in-domain
        train, val, test = gen_in_domain_data(df,
                                              train_domain,
                                              test_domain,
                                              train_frac=train_frac,
                                              seed=seed)
    else:
        # out-of-domain
        train, val, test = gen_out_domain_data(df,
                                               train_domain,
                                               test_domain,
                                               train_frac=train_frac,
                                               seed=seed)

    dataset_dict = {'train': train, 'val': val, 'test': test}

    # split train/val/test into source and target domains
    if semi_supervised:
        dataset_dict = gen_semi_supervised(train,
                                           val,
                                           test,
                                           train_frac=train_frac,
                                           seed=seed)

    # init the tokenzier
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              use_fast=True,
                                              local_files_only=True)
    # encode the text and take the labels
    datasets = {}
    for k, v in dataset_dict.items():
        datasets[k] = create_MFData(v, tokenizer, mf_label_names, max_seq_len)

    return datasets
