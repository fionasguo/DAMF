"""
Functions for inference - predict and evaluate
"""

import os
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, List

from model import MFBasic, MFDomainAdapt
from data_loader import MFData


def compute_feat(
        model: torch.nn.Module,
        dataset: MFData,
        device: str,
        batch_size: int = 64) -> Tuple[List, List]:
    """
    Compute feature embeddings (both pooler outputs and last hidden states) for the given dataset using the given model.

    Args:
        model: MFBasic or MFDomainAdapt
        dataset: test data, an MFData instance
        device: cpu or gpu
        batch_size: default is 64
    Returns:
        feature embeddings
        domain labels
    """
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False)

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    model.eval()

    i = 0

    feats = []
    last_hidden_states = []
    domain_labels = []

    while i < len_dataloader:
        data, _ = data_iter.next()
        for k, v in data.items():
            data[k] = data[k].to(device)

        with torch.no_grad():
            try:
                last_hidden_state, feat = model.module.gen_feature_embeddings(
                    data['input_ids'], data['attention_mask'])
            except:
                last_hidden_state, feat = model.gen_feature_embeddings(
                    data['input_ids'], data['attention_mask'])

        feat = feat.detach()
        feats.extend(feat.data.tolist())

        last_hidden_state = last_hidden_state.detach()
        last_hidden_states.extend(last_hidden_state.to('cpu').tolist())

        domain_labels.extend(data['domain_labels'].tolist())

        i += 1

    dataset.feat_embed = last_hidden_states

    return feats, domain_labels


def plot_heatmap(
        feats: List,
        domain_labels: List,
        fig_save_path: str):
    """
    Plot feature embeddings as heatmap.

    Can be used to see if source and target domain have distinct features.
    """
    plt.figure(figsize=[10, 8])
    fig = sns.heatmap(feats, yticklabels=domain_labels).get_figure()

    fig.savefig(fig_save_path + '/heatmap.png', format='png')


def plot_tsne(
        feats: List,
        domain_labels: List,
        fig_save_path: str):
    """
    Plot feature embeddings as tsne.

    Can be used to see if source and target domain have distinct features.
    """
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 8))
    plt.xlim((-40, 40))
    plt.ylim((-40, 40))
    fig = sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=domain_labels,
        palette=sns.color_palette("hls", len(np.unique(domain_labels))),
        legend="full",
        alpha=0.3
    ).get_figure()

    fig.savefig(fig_save_path + '/tsne.png', format='png')


def feature_embedding_analysis(
        source_dataset: MFData,
        target_dataset: MFData,
        fig_save_path: str,
        batch_size: int = 64,
        model: torch.nn.Module = None,
        model_path: str = None):
    """
    Plot feature embeddings as heatmaps and tsne.

    Can be used to see if source and target domain have distinct features.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert (model != None or model_path !=
            None), 'Provide a model object or a model path.'
    if model == None:
        model = torch.load(model_path, map_location=torch.device(device))

    model = model.eval()

    model = model.to(device)

    # get feature embeddings of source and target data from the best model (adv model)
    s_feats, s_domain_labels = compute_feat(
        model, source_dataset, batch_size, device)
    t_feats, t_domain_labels = compute_feat(
        model, target_dataset, batch_size, device)

    s_feats.extend(t_feats)
    s_domain_labels.extend(t_domain_labels)
    feats = np.array(s_feats)
    domain_labels = np.array(s_domain_labels).squeeze()

    np.savetxt(fig_save_path + '/feat_embeddings.tsv', feats)
    np.savetxt(fig_save_path + '/domain_labels.tsv', domain_labels)

    plot_heatmap(feats, domain_labels, fig_save_path)
    plot_tsne(feats, domain_labels, fig_save_path)
