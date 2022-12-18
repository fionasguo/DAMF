import os
import random
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
import logging
from transformers import BertModel,BertTokenizer


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

correct_count = Counter()
chosen_count = Counter()

def predict(model, dev, mf_classes, device):
    model.eval()

    features = torch.stack(dev['embeddings'].tolist()).squeeze().to(device)
    labels = dev[mf_classes].values.tolist()

    preds = []

    batch_size = 64

    for b in range(0, len(features), batch_size):
        with torch.no_grad():
            outputs = model(features[b:b+batch_size])

        pred_confidence = torch.sigmoid(outputs)
        pred = ((pred_confidence) >= 0.5).long()
        preds.extend(pred.to('cpu').tolist())

    return preds, labels

def train_classifier(train,mf_classes,device):
	# init model
    model = torch.nn.Linear(768,len(mf_classes))
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters())
    # loss function
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

    batch_size = 64

    for epoch in range(3):
        model.train()
        # data
        train = train.sample(frac=1, random_state=2022+epoch)
        features = torch.stack(train['embeddings'].tolist()).squeeze().to(device)
        labels = torch.tensor(train[mf_classes].values, dtype=torch.float).to(device)

        for b in range(0, len(features), batch_size):
            model.zero_grad()

            outputs = model(features[b:b+batch_size])
            loss = loss_fn(outputs, labels[b:b+batch_size])

            loss.backward()
            optimizer.step()

    return model

def run_iteration(data, mf_classes, target_size, device):
    global correct_count, chosen_count

    # partition into train and dev
    train_size = target_size
    train = data.sample(n=train_size)
    dev = data.drop(train.index)

    # train
    model = train_classifier(train,mf_classes,device)
    # predict on dev set
    mf_preds, mf_labels = predict(model, dev, mf_classes, device)
    dev['is_pred_correct'] = np.all(np.array(mf_preds) == np.array(mf_labels), axis=1)
    # update counts
    chosen_count.update(list(dev.index))
    correct_ids = list(dev.index[dev.is_pred_correct==True])
    # correct_ids = [dev_sample_ids[sid] for sid in range(len(dev_sample_ids)) if preds[sid]]
    correct_count.update(correct_ids)

def run_aflite(data, mf_classes, target_frac=0.5, max_seq_len=50):
    """
    Args:
        data: pandas DataFrame, must have column 'text' and columns with names in mf_classes
        mf_classes: a list of mf class names
        target_frac: reduce data size to this target fraction
        max_seq_len: max length for bert tokenizer

    Return:
        filtered data with a smaller size
    """
    data = data.reset_index()
    logging.info('Running AFLite on data with original size of %d, reducing to target fraction of %f', len(data), target_frac)

    # compute vanilla bert embeddings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    def embed_text(text, model, tokenizer, device):
        encodings = tokenizer(text,truncation=True, max_length=max_seq_len, padding="max_length", return_tensors='pt')
        with torch.no_grad():
            encodings = encodings.to(device)
            output = model(**encodings).pooler_output
        return output

    data['embeddings'] = data['text'].apply(lambda x: embed_text(x, model, tokenizer, device))

    # set target size
    target_size = int(len(data)*target_frac)
    cutoff_size = int(len(data)*0.02)

    global correct_count, chosen_count
    while len(data) > target_size:
        correct_count = Counter()
        chosen_count = Counter()

        for i in tqdm(range(64)):
            set_seed(i)
            run_iteration(data, mf_classes, target_size, device)

        for k, v in correct_count.items():
            correct_count[k] = float(v)/chosen_count[k]
        sorted_correct_count = sorted(correct_count.items(), key=lambda x: x[1], reverse=True)
        easy_train = [s[0] for s in sorted_correct_count[:cutoff_size] if s[1] > 0.75]
        data = data.drop(easy_train,axis=0)
        newly_removed = len(easy_train)

        logging.debug('now keeping %d', len(data))
        if newly_removed < cutoff_size:
            break

    print ('finally keeping %d', len(data))

    return data
