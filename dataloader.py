#%%
from typing import Sequence
from numpy.lib.ufunclike import fix
import torch
from torch.autograd.grad_mode import F
import torch.nn as nn
import torch.optim as optim
import os
from torchtext.legacy import data, datasets
from torchtext.legacy.data.dataset import Dataset
from torchtext.legacy.data.iterator import batch

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field,BucketIterator
from spacy.lang.fr.examples import sentences 

import spacy
import numpy as np
import random
import math
import time

from torchtext.legacy.datasets.translation import TranslationDataset

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# 각 언어에 맞는 tokenizer 불러오기 
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')
# %%
def tokenize_de(text):
    # 독일어 tokenize해서 단어들을 리스트로 만든 후 reverse 
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
    
def tokenize_en(text):
    # 영어 tokenize해서 단어들을 리스트로 만들기
    return [tok.text for tok in spacy_en.tokenizer(text)]

#%%

class DataLoader():
    def __init__(self,
                batch_size,
                device=device,
                max_vocab=9999999,
                max_length=255,
                fix_length=None,
                shuffle=True
                ):
        super(DataLoader,self).__init__()
        

        self.src=data.Field(use_vocab=True,
                            tokenize = tokenize_de,
                            fix_length=fix_length,
                            init_token='<sos>',
                            eos_token='<eos>', 
                            lower=True
                            )

        self.trg=data.Field(use_vocab=True,
                            tokenize=tokenize_en,
                            fix_length=fix_length,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True
                            )

        train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(self.src,self.trg))

           
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size,device=device )


        self.src.build_vocab(train_data, max_size = max_vocab)
        self.trg.build_vocab(train_data, max_size = max_vocab)

    def load_vocab(self,src_vocab,trg_vocab):
        self.src.vocab=src_vocab
        self.trg.vocab=trg_vocab
        
