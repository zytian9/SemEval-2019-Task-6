# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torchtext import data
from torchtext.data import Field
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
from tqdm import tqdm, tqdm_notebook, tnrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = r'./training-v1/offenseval-training-v1.tsv'
trial_path = r'./trial-data/offenseval-trial-v2.tsv'
text = pd.read_csv(path,sep="\t")
trial = pd.read_csv(trial_path,sep="\t")


# Task a preprocessing
x = text["tweet"]
text['subtask_a'] = np.where(text["subtask_a"]=="OFF",1,0)
text.to_csv('././training-v1/offenseval-training-v2.tsv',sep='\t',index=False)

trial2 = pd.DataFrame()
trial2["tweet"] = trial.iloc[:,0]
trial2['subtask_a'] = np.where(trial["NOT"]=="OFF",1,0)
trial2.to_csv('./trial-data/offenseval-trial-v2.tsv',sep='\t',index=False)
# inspect data
fig = plt.figure(figsize=(8,5))
ax = sns.barplot(x=y.unique(),y=y.value_counts());
ax.set(xlabel='Labels')


# begin segmentation
nlp = spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner'])

# create a tokenizer function
def tokenizer(text): 
    return [tok.text for tok in nlp.tokenizer(text)]

# define preproessing pipline
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True,include_lengths=True, use_vocab=True)
LABEL = data.Field(sequential=False,use_vocab=False)

# load .tsv data and generate tokenized trainign data
from torchtext.data import TabularDataset
 
tv_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("tweet", TEXT), ("subtask_a", LABEL),
                 ("subtask_b", None), ("subtask_c", None)]

trn = TabularDataset(
               path='./training-v1/offenseval-training-v2.tsv', # the root directory where the data lies
               format='tsv',
               skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
               fields=tv_datafields)

trn[0].__dict__.keys()
trn[0].tweet
trn[0].subtask_a




# load .tsv data and generate tokenized test data
test_datafields = [("tweet", TEXT), ("subtask_a", LABEL)]

tst = TabularDataset(
               path='./trial-data/offenseval-trial-v2.tsv', # the root directory where the data lies
               format='tsv',
               skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
               fields=test_datafields)

tst[0].__dict__.keys()
tst[0].tweet
tst[0].subtask_a

# use embeddings
TEXT.build_vocab(trn,tst,max_size=100000,vectors='glove.twitter.27B.100d')
#LABEL.build_vocab(trn, tst)


# construct the iterator
from torchtext.data import Iterator, BucketIterator



traindl, valdl = data.BucketIterator.splits(datasets=(trn, tst), 
                                            batch_sizes=(64,64), 
                                            sort_key=lambda x: len(x.tweet), 
                                            device=device, 
                                            sort_within_batch=True, 
                                            repeat=False)



# define batch generator

class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)
            
train_batch_it = BatchGenerator(traindl, 'tweet', 'subtask_a')
print(next(iter(train_batch_it)))


# define model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

vocab_size = len(TEXT.vocab)
embedding_dim = 100
n_hidden = 256
n_out = 2


class ConcatPoolingGRUAdaptive(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out, pretrained_vec, bidirectional=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bidirectional = bidirectional
        
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb.weight.data.copy_(pretrained_vec) # load pretrained vectors
        self.emb.weight.requires_grad = False # make embedding non trainable
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden, bidirectional=bidirectional)
        if bidirectional:
            self.out = nn.Linear(self.n_hidden*2*2, self.n_out)
        else:
            self.out = nn.Linear(self.n_hidden*2, self.n_out)
        
    def forward(self, seq, lengths):
        bs = seq.size(1)
        self.h = self.init_hidden(bs)
        seq = seq.transpose(0,1)
        embs = self.emb(seq)
        embs = embs.transpose(0,1)
        embs = pack_padded_sequence(embs, lengths)
        gru_out, self.h = self.gru(embs, self.h)
        gru_out, lengths = pad_packed_sequence(gru_out)        
        
        avg_pool = F.adaptive_avg_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)
        max_pool = F.adaptive_max_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)        
#         outp = self.out(torch.cat([self.h[-1],avg_pool,max_pool],dim=1))
        outp = self.out(torch.cat([avg_pool,max_pool],dim=1))
        return F.log_softmax(outp, dim=-1)
    
    def init_hidden(self, batch_size): 
        if self.bidirectional:
            return torch.zeros((2,batch_size,self.n_hidden)).cuda().to(device)
        else:
            return torch.zeros((1,batch_size,self.n_hidden)).cuda().to(device)

m = ConcatPoolingGRUAdaptive(vocab_size, embedding_dim, n_hidden, n_out, 
                             trn.fields['tweet'].vocab.vectors).to(device)

from sklearn.metrics import accuracy_score

train_batch_it = BatchGenerator(traindl, 'tweet', 'subtask_a')
val_batch_it = BatchGenerator(valdl, 'tweet', 'subtask_a')
opt = optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), 3e-4)
loss_fn=F.nll_loss
epochs=5
num_batch = len(train_batch_it)
for epoch in tnrange(epochs):      
    y_true_train = list()
    y_pred_train = list()
    total_loss_train = 0          
    
    t = tqdm(iter(train_batch_it), leave=False, total=num_batch)
    for (X,lengths),y in t:
        t.set_description(f'Epoch {epoch}')
        lengths = lengths.cpu().numpy()
        
        opt.zero_grad()
        pred = m(X, lengths)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        
        t.set_postfix(loss=loss.item())
        pred_idx = torch.max(pred, dim=1)[1]
        
        y_true_train += list(y.cpu().data.numpy())
        y_pred_train += list(pred_idx.cpu().data.numpy())
        total_loss_train += loss.item()
        
    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_loss = total_loss_train/len(train_batch_it)
    
    if val_batch_it:
        y_true_val = list()
        y_pred_val = list()
        total_loss_val = 0
        for (X,lengths),y in tqdm(val_batch_it, leave=False):
            pred = m(X, lengths.cpu().numpy())
            loss = loss_fn(pred, y)
            pred_idx = torch.max(pred, 1)[1]
            y_true_val += list(y.cpu().data.numpy())
            y_pred_val += list(pred_idx.cpu().data.numpy())
            total_loss_val += loss.item()
        valacc = accuracy_score(y_true_val, y_pred_val)
        valloss = total_loss_val/len(val_batch_it)
        print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {valloss:.4f} val_acc: {valacc:.4f}')
    else:
        print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}')
        
                
