"""
Read train/dev/test datasets from a csv file (data_dir).

data_dir can either be:
- a csv path, where train/val/test are all together and mf labels are present in all data examples, and the functions will split them. This is mostly for testing models.
- a dir where separate csv files exist for train/val/test data. Some sets might be missing (eg test) in this case.

The data csv needs to have these columns: 'text','domain' and all columns in desired mf_label_names.
"""

import os
import pandas as pd
from typing import List, Tuple, Dict

from transformers import AutoTokenizer

from .preprocessing import preprocess_tweet
from .data_loader import MFData
from .train_test_split import gen_in_domain_data, gen_out_domain_data, gen_semi_supervised

pd.options.mode.chained_assignment = None


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

    # some dataset might not have mf labels, eg t_train or test data
    if any([l not in df.columns for l in mf_label_names
            ]) or df[mf_label_names].isnull().values.any():
        # if any of the mf labels is not in the data columns, or if any of the mf label columns has NaN values
        mf_labels = None
    else:
        mf_labels = df[mf_label_names].values

    domain_labels = df['domain_idx'].values

    mf_data = MFData(encodings, mf_labels, domain_labels)

    return mf_data


def train_test_split(data_dir: str,
                     train_domain: List[str],
                     test_domain: List[str],
                     semi_supervised: bool,
                     train_frac: float = 0.8,
                     seed: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Generate train/dev/test datasets.

    Args:
        data_dir: should be a csv with these columns: 'text','domain' and mf_label_names
        train_domain: the name of the domains for training, should be in the 'domain' column in df
        test_domain: the name of the domains for testing
        semi_supervised: if true, separate train/dev/test data by source and target domains
        train_frac: the fraction of number of data points used for training
        seed: random seed

    Returns:
        dictionary of train, val, test df objects
    """
    # read data
    df = pd.read_csv(data_dir)
    # preprocess
    df.text = df.text.apply(preprocess_tweet)
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

    return dataset_dict


def load_data(data_dir: str, train_domain: List[str],
              test_domain: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Directly load data from files.
    """
    dataset_dict = {}
    domains = list(set(train_domain).union(set(test_domain)))
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
        data_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(os.path.join(data_dir, file))

        df['domain_idx'] = df['domain'].apply(lambda x: domains.index(x))

        df.text = df.text.apply(preprocess_tweet)
        df = df.reset_index()

        dataset_dict[data_name] = df

    return dataset_dict


def read_data(data_dir: str,
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

    # if data_dir is a directory, directly load all the files
    if os.path.isdir(data_dir):
        dataset_dict = load_data(data_dir, train_domain, test_domain)
    else:
        # if data_dir is a csv, peform train test split
        dataset_dict = train_test_split(data_dir, train_domain, test_domain,
                                        semi_supervised, train_frac, seed)

    # init the tokenzier
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              use_fast=True,
                                              local_files_only=True)
    # encode the text and take the labels
    datasets = {}
    for k, v in dataset_dict.items():
        datasets[k] = create_MFData(v, tokenizer, mf_label_names, max_seq_len)

    return datasets
