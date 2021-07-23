import math
from datasets import load_dataset
from math_timu import idx2label
import pickle
import pandas as pd
import numpy as np
import os
from os.path import join, dirname, abspath
PROJECT_PATH = join(dirname(dirname(abspath(__file__))))
import sys
sys.path.append(PROJECT_PATH)
from tqdm import tqdm
import gensim
import json
from pprint import pprint
from torch.utils.data import DataLoader
from itertools import chain

import tez
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

import matplotlib.pyplot as plt
from sklearn import metrics, model_selection, preprocessing
from optim import AdamW
from self_transformer import TransformerEncoder, TransformerEncoderLayer

from YiQiSegger import YiQiSegger

go_emotions = load_dataset(path='./math_timu.py')

data = go_emotions.data

train:pd.DataFrame = go_emotions.data["train"].to_pandas()
valid:pd.DataFrame = go_emotions.data["validation"].to_pandas()
test:pd.DataFrame = go_emotions.data["test"].to_pandas()

print(train.head(10))

n_labels= len(idx2label)
model_path = 'model_ckpt/model.bin'
os.makedirs(dirname(model_path), exist_ok=True)

def one_hot_labels(df, name):
    if os.path.exists(name):
        return pd.read_pickle(name)
    dict_labels = []
    for i in tqdm(range(len(df)), leave=False):
        d = dict(zip(range(n_labels), [0]*n_labels))
        labels = df.loc[i]["labels"]
        for label in labels:
            d[label] = 1
        dict_labels.append(d)
    df_labels = pd.DataFrame(dict_labels)
    pd.to_pickle(df_labels, name)
    return df_labels

train_oh_labels = one_hot_labels(train, 'caches/train_labels.df')
valid_oh_labels = one_hot_labels(valid, 'caches/valid_labels.df')
test_oh_labels = one_hot_labels(test, 'caches/test_labels.df')

class Text:
    def __init__(self):
        self.tokenizer = YiQiSegger()
        self.special_tok = ['<pad>']
        self.tok2idx = {}
        self.idx2tok = {}
        self.token_idx = 0
        self.max_len = None
        self.cache_path = './caches/Text.pkl'
        self.w2v_path = './caches/word2vec.model'
        self.corpus_path = './caches/corpus.txt'

    def build_vocab(self):
        # load from disk
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.tok2idx, self.idx2tok, self.token_idx, self.max_len = pickle.load(f)
            return

        # 从corpus中生成vocab
        def corpus_tokens():
            with open(self.corpus_path) as f:
                for line in f.readlines():
                    yield line.strip()

        def corpus_tokens2():
            for line in chain(train.text, valid.text, test.text):
                yield line

        for tok in self.special_tok:
            if tok not in self.tok2idx:
                self.tok2idx[tok] = self.token_idx
                self.idx2tok[self.token_idx] = tok
                self.token_idx += 1
        seq_lengths = []
        for tokens in corpus_tokens2():
            if isinstance(tokens, str):
                tokens = self.tokenize(tokens)
            seq_lengths.append(len(tokens))
            for tok in tokens:
                if tok not in self.tok2idx:
                    self.tok2idx[tok] = self.token_idx
                    self.idx2tok[self.token_idx] = tok
                    self.token_idx += 1

        assert len(self.tok2idx) == len(self.idx2tok)
        self.max_len = int(np.percentile(seq_lengths, 95))

        # pickl dump to disk
        with open(self.cache_path, 'wb') as f:
            pickle.dump([self.tok2idx, self.idx2tok, self.token_idx, self.max_len], f)

    def load_gensim_vec(self):
        vecs = []
        w2v_model = gensim.models.Word2Vec.load(self.w2v_path)
        for tok, idx in self.tok2idx.items():
            if tok in w2v_model.wv.vocab:
                vecs.append(torch.tensor(w2v_model.wv.get_vector(tok), dtype=torch.float32).unsqueeze(0))
            else:
                vecs.append(torch.zeros(1, w2v_model.wv.vector_size, dtype=torch.float32))
        return torch.cat(vecs, dim=0)

    def tokenize(self, s):
        tokens = self.tokenizer.token_for_kps(s)
        return tokens

    def encode(self, s):
        if isinstance(s, list):
            return [self.tok2idx.get(tok, 0) for tok in s][:self.max_len]
        else:
            return [self.tok2idx.get(tok, 0) for tok in self.tokenize(s)][:self.max_len]

TEXT = Text()
TEXT.build_vocab()

class GoEmotionDataset():
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        target = self.targets[index]
        text = self.texts[index]

        ids = TEXT.encode(text)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.long),
            "text":text,
        }
def collate_fn(x:list):
    ids = [i['ids'] for i in x]
    targets = [i['targets'] for i in x]
    texts = [i['text'] for i in x]

    seq_length = torch.tensor([s.size(0) for s in ids], dtype=torch.long)
    ids = pad_sequence(ids, batch_first=True)
    targets = torch.stack(targets, dim=0)
    return {'ids':ids, 'seq_len':seq_length, 'targets':targets}# , "texts":texts}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EmotionClassifier(tez.Model):
    def __init__(self, num_train_steps, num_classes):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(TEXT.load_gensim_vec())
        # w2v_model = gensim.models.Word2Vec.load('word2vec.model')
        self.pos_encoder = PositionalEncoding(d_model=self.embed.embedding_dim)
        self.encoder = TransformerEncoder(TransformerEncoderLayer(self.embed.embedding_dim, nhead=8, dim_feedforward=1024), num_layers=4)
        self.out = nn.Linear(self.embed.embedding_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=1e-4)
        return opt

    def fetch_scheduler(self):
        # sch = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        # )
        # sch = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        return None

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets.float())

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}

        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        topk_recall = report2(outputs, targets)

        fpr_micro, tpr_micro, _ = metrics.roc_curve(targets.ravel(), outputs.ravel())
        auc_micro = metrics.auc(fpr_micro, tpr_micro)
        return {"topk_recall": topk_recall, 'auc':auc_micro}

    def forward(self, ids, seq_len, targets=None):
        embedded = self.embed(ids)
        embedded = self.pos_encoder(embedded)
        mask = torch.where(ids == 0, 1, 0).to(torch.bool)
        
        embedded = torch.transpose(embedded, 0, 1)
        hidden = self.encoder(embedded, src_key_padding_mask=mask)
        hidden = torch.mean(hidden, dim=0)
        output = self.out(hidden)
        output = self.dropout(output)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc

def get_labels(array):
    labels = []
    for row in array.tolist():
        tmp = []
        for idx, col in row:
            if col == 1:
                tmp.append(idx)
        labels.append(tmp[:])
    return labels


def get_labels_from_onhot(onhot_array):
    target = []
    for row in onhot_array.tolist():
        tmp = []
        for idx, col in enumerate(row):
            if col == 1:
                tmp.append(idx)
        target.append(tmp[:])
    return target

def report(outputs, targets):
    to_save = []
    threshholds = np.linspace(0, 1, 100)
    for thresh in threshholds:
        pred = np.where(outputs >= thresh, 1, 0)
        pred = pred.astype('int')

        true_nums = np.count_nonzero(targets)
        pred_nums = np.count_nonzero(pred)
        equal_num = np.count_nonzero(np.bitwise_and(pred, targets))
        union_nums = np.count_nonzero(np.bitwise_or(pred, targets))
        strick_equal = np.bitwise_xor(pred, targets).sum(axis=-1)
        strick_equal = np.where(strick_equal > 0, 0, 1)
        try:
            precision = equal_num / pred_nums
            recall = equal_num / true_nums
            iou = equal_num / union_nums
            exact_equal = sum(strick_equal) / len(strick_equal)
            to_save.append({'thresh':thresh,
                            'precision':precision,
                            'recall':recall,
                            'iou':iou,
                            'exact_equal':exact_equal,
                            'f1': 2*precision*recall / (precision + recall)
                           }
                           )
        except ZeroDivisionError:
            pass


    print('iou最高:\n')
    pprint(sorted(to_save, key=lambda x:x['iou'], reverse=True)[0])
    print('f1 最高:\n')
    pprint(sorted(to_save, key=lambda x:x['f1'], reverse=True)[0])
    print('exact_equal最高:\n')
    pprint(sorted(to_save, key=lambda x:x['exact_equal'], reverse=True)[0])
    with open('test_output/test_metric.json', 'w') as f:
        json.dump(to_save, f, indent=4)

    return

def report2(outputs, test_oh_labels, topk=5, stage='train'):
    target = get_labels_from_onhot(test_oh_labels)
    preds = np.argsort(outputs)[:, -topk:].tolist()  
    records = []
    for i in range(len(target)):
        if target[i] and not (set(target[i]) -set( preds[i])):
            records.append(1)
        else:
            records.append(0)
    recall = sum(records)/len(records)
    if stage == 'test':
        print(f'recall of top {topk}: ', recall)
    return recall
    
def label_prob(texts, pred_probs, onhot_labels, name):
    labels = get_labels_from_onhot(onhot_labels)
    result = []
    for i in range(len(texts)):
        text = texts[i]
        label = labels[i]
        pred_prob = pred_probs[i]
        pred_top_label = np.argsort(pred_prob)[::-1][:3]
        pred_top_label = [idx2label[i] for i in pred_top_label]
        label_prob = [(idx2label[i], float(pred_prob[i])) for i in label]
        tmp = {"text":text, "label_prob":label_prob, "pred_topk":pred_top_label}
        result.append({**tmp})

    with open(f'test_output/{name}_label_prob.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)



def model_test(model:nn.Module, name, device='cpu' ):
    output_cache = f'./caches/{name}_outputs.torch'
    text_cache = f'./caches/{name}_texts.list'
    model.load(model_path, device=device)
    model.eval()
    model = model.to(device)

    if os.path.exists(output_cache) and os.path.exists(text_cache):
        outputs = torch.load(open(output_cache, 'rb'))
        with open(text_cache, 'rb') as f:
            texts = pickle.load(f)

    else:
        if name == 'test':
            test_data = test
            label_data = test_oh_labels
        elif name == 'valid':
            test_data = valid
            label_data = valid_oh_labels
        elif name == 'train':
            test_data = train
            label_data = train_oh_labels

        test_dataset = GoEmotionDataset(test_data.text.tolist(), label_data.values.tolist())
        dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=collate_fn)

        outputs = []
        texts = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                output, loss, acc = model.forward(batch["ids"].to(device),
                                                  batch["seq_len"].to("cpu"),
                                                  batch["targets"].to(device)
                                                  )
                outputs.append(output)
                # texts.extend(batch['texts'])

        outputs = torch.cat(outputs)
        outputs = torch.sigmoid(outputs)
        with open(output_cache, mode='wb') as f:
            torch.save(outputs, f)
        with open(text_cache, 'wb') as f:
            pickle.dump(texts, f)

    outputs = outputs.cpu().detach().numpy()
    
    report(outputs, test_oh_labels.values)
    report2(outputs, test_oh_labels.values, topk=3, stage='test')
    report2(outputs, test_oh_labels.values, stage='test')
    report2(outputs, test_oh_labels.values, topk=7, stage='test')
    report2(outputs, test_oh_labels.values, topk=9, stage='test')

    # label_prob(texts, outputs, train_oh_labels.values, name=name)



    def score_sentence(text, topn=5):
        with torch.no_grad():
            ids = TEXT.encode(text)
            seq_len = torch.tensor([len(ids)])

            ids = torch.LongTensor(ids).unsqueeze(0)
            ids = ids.to(device)

            output, _, _ = model.forward(ids, seq_len)
            output = torch.sigmoid(output)

            probas, indices = torch.sort(output)

        probas = probas.cpu().numpy()[0][::-1]
        indices = indices.cpu().numpy()[0][::-1]

        print('example: ', text, '\n')
        for i, p in zip(indices[:topn], probas[:topn]):
            print(idx2label[i], p)

    score_sentence("3.  我们组有  7人。  我们组的人数和  你们组同样多。  两组一共有多少人？  \_\$\$\_ \_\$\$\_ \_\$\$\_ = \_\$\$\_ ( 人 )")

if __name__ == '__main__':
    train_stage = True
    if train_stage:
        train_dataset = GoEmotionDataset(train.text.tolist(), train_oh_labels.values.tolist())
        valid_dataset = GoEmotionDataset(valid.text.tolist(), valid_oh_labels.values.tolist())

        n_train_steps = int(len(train) / 32 * 10)


        tb_logger = tez.callbacks.TensorBoardLogger(log_dir="tf_logs")
        es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path=model_path, delta=0.0)
        model = EmotionClassifier(n_train_steps, n_labels)
        model.train()
        #model.load('export2/model.bin')
        model.fit(train_dataset,
                  valid_dataset,
                  train_bs=128,
                  device="cuda",
                  epochs=120,
                  callbacks=[tb_logger, es],
                  fp16=True,
                  n_jobs=0,
                  train_collate_fn=collate_fn,
                  valid_collate_fn=collate_fn,
                  )
        model_test(model.eval(), 'test','cuda')
    else:
        model = EmotionClassifier(None, n_labels)

        model_test(model, 'test',device='cuda')


