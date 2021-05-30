
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchtext.legacy.data.iterator import batch
#%%
class Trainer(object):
    def __init__(self, args, loaders, model):
        self.args = args
        self.model = model
        self.loaders = loaders
        trg_pad_idx = self.loaders.trg.vocab.stoi[self.loaders.trg.pad_token]
        self.device = torch.device('cuda')
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def train(self, data_loader): 
        epoch_loss= 0
        self.model.train() # train mode

        for i, batch in enumerate(data_loader): # index, data # 위에 출력한거 확인! i=train loader의 길이-1
            src=batch.src
            trg=batch.trg

            self.optimizer.zero_grad()
            output=self.model(src,trg)
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim=output.shape[-1]

            # loss 함수는 2d input으로만 계산 가능 
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
        
            # trg = [(trg len-1) * batch size]
            # output = [(trg len-1) * batch size, output dim)]
            loss = self.criterion(output, trg)
        
            loss.backward()
        
        
            self.optimizer.step()
        
            epoch_loss+=loss.item()
        
        return epoch_loss/len(data_loader)

    def evaluate(self, data_loader):
        epoch_loss=0
        self.model.eval() # evaluation mode

        with torch.no_grad(): #backprop 기능 끄기
            for i, batch in enumerate(data_loader):
                src = batch.src
                trg = batch.trg
            
                # teacher_forcing_ratio = 0 (아무것도 알려주면 안 됨)
                output = self.model(src, trg, 0)
            
            
                output_dim = output.shape[-1]
            
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
            
            
            
                loss = self.criterion(output, trg)
            
                epoch_loss+=loss.item()
        
            return epoch_loss/len(data_loader)
    
    

