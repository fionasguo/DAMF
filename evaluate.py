"""
Functions for inference - predict and evaluate
"""

import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import logging
from typing import Tuple, List

from model import MFBasic, MFDomainAdapt
from data_loader import MFData


def predict(model: torch.nn.Module,
            dataset: MFData,
            device: str,
            batch_size: int = 64,
            domain_adapt: bool = True,
            is_adv: bool = True) -> Tuple[List, List, List, List]:
    """
    Predict MF and/or domain labels based on given model.

    Args:
        model: MFBasic or MFDomainAdapt
        dataset: test data, an MFData instance
        device: cpu or gpu
        batch_size: default is 64
        domain_adapt: whether using basic or domain adapt model
        is_adv: if doing adv training, will pass it to model forward fn and predict domain labels
    Returns:
        mf_preds, mf_labels: moral foundation predictions and true labels
        domain_preds, domain_labels: domain predictions and true labels
    """
    model.eval()

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False)
    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0

    mf_preds = []
    mf_labels = []
    domain_preds = []
    domain_labels = []

    while i < len_dataloader:
        data_target, _ = data_target_iter.next()
        for k, v in data_target.items():
            data_target[k] = data_target[k].to(device)

        with torch.no_grad():
            if not domain_adapt:
                outputs = model(data_target['input_ids'],
                                data_target['attention_mask'])
            else:
                outputs = model(data_target['input_ids'],
                                data_target['attention_mask'],
                                data_target['domain_labels'],
                                lambda_domain=0,
                                adv=is_adv)

        mf_pred_confidence = torch.sigmoid(outputs['class_output'])
        mf_pred = ((mf_pred_confidence) >= 0.5).long()
        mf_preds.extend(mf_pred.to('cpu').tolist())
        mf_labels.extend(data_target['mf_labels'].to('cpu').tolist())

        if domain_adapt and is_adv:
            domain_pred = outputs['domain_output'].data.max(1, keepdim=True)[1]
            domain_preds.extend(domain_pred.to('cpu').tolist())
            domain_labels.extend(
                data_target['domain_labels'].to('cpu').tolist())

        i += 1

    return mf_preds, mf_labels, domain_preds, domain_labels


def evaluate(dataset: MFData,
             batch_size: int = 64,
             model: torch.nn.Module = None,
             model_path: str = None,
             is_adv: bool = True,
             test: bool = False) -> float:
    """
    Evalute test data and print F1 scores.

    Args:
        dataset: test data, an MFData instance
        batch_size: default is 64
        model: MFBasic or MFDomainAdapt instance, either model or model path should be given
        model_path: if no model instance is given, will load model from this path
        is_adv: if doing adv training, will pass it to model forward fn and predict domain labels
        test: whether in training or test mode

    Returns:
        f1 score
        also print detailed classification report to log file.
    """
    if test:
        logger = logging.getLogger(__name__)
        orig_level = logger.level
        logger.setLevel(logging.DEBUG)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert (model != None
            or model_path != None), 'Provide a model instance or a model path.'
    if model == None:
        model = torch.load(model_path, map_location=torch.device(device))

    # for evaluation, don't use nn.DataParallel (multiple gpus)
    try:
        model = model.module.to(device)
    except:
        model = model.to(device)

    # check whether if the model is domain adapt model
    domain_adapt = (isinstance(model, MFDomainAdapt))

    # predict
    mf_preds, mf_labels, domain_preds, domain_labels = predict(
        model, dataset, device, batch_size, domain_adapt, is_adv)

    # print reports
    mf_report = metrics.classification_report(mf_labels,
                                              mf_preds,
                                              zero_division=0)
    logging.debug('MF classification report:')
    logging.debug(mf_report)
    conf_matrix = metrics.multilabel_confusion_matrix(mf_labels, mf_preds)
    logging.debug('MF classification confusion matrix:')
    logging.debug(conf_matrix)
    mf_report = metrics.classification_report(mf_labels,
                                              mf_preds,
                                              zero_division=0,
                                              output_dict=True)
    macro_f1 = mf_report['weighted avg']['f1-score']

    if domain_adapt and is_adv:
        domain_report = metrics.classification_report(domain_labels,
                                                      domain_preds,
                                                      zero_division=0)
        conf_matrix = metrics.confusion_matrix(domain_labels, domain_preds)
        logging.debug('Domain classification report:')
        logging.debug(domain_report)
        logging.debug('Domain classification confusion matrix:')
        logging.debug(conf_matrix)

    # set logging level back to original
    if test:
        logger.setLevel(orig_level)

    return macro_f1
