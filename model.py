import torch
import torch.nn as nn
import torch.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim,dropout=0.5):
        super().__init__()
        
        self. hid_dim = hid_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src_len,batch_size]
        embedded = self.dropout(self.embedding(src))
        
        # embedded = [len,batch size,emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        #output=[len, batch size, hid dim]
        #hidden=[1,batch,hid dim]
        #cell=[1,batch,hid dim]
        
        return outputs,hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim,  emb_dim, hid_dim, dropout=0.5):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [1,batch, hid dim]
        # cell = [  1,batch, hid dim]
   
 
        # context = [1,batch,  hid dim]
        
        # input = [1,batch]
        input = input.unsqueeze(0)
        
        # embedded = [  1,batch size, emb dim]
        embedded = self.dropout(self.embedding(input))
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # output = [1,batch,hid dim]
        # hidden = [1,batch,hid dim]
        # cell = [1,batch, hid dim]
        
        
        
        # prediction = [batch size, output dim]
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden




class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.device=device

        #src=[src_len,batch]
        #trg=[trg_len,batch]
    def forward(self,src,trg,teacher_forcing_ratio=0.5):
        trg_len=trg.shape[0]
        batch_size=trg.shape[1]
        trg_vocab_size=self.decoder.output_dim

        # decoder 결과를 저장할 텐서
        outputs = torch.zeros( trg_len,batch_size,trg_vocab_size).to(self.device)

       
        enc_outputs,hidden,cell = self.encoder(src)

        # Decoder에 들어갈 첫 input은 <sos> 토큰
        input = trg[0,:]
       
        # target length만큼 반복

        for i in range(1,trg_len):
            output, hidden= self.decoder(input, hidden,cell)
            outputs[i] = output
            teacher_force = random.random() < teacher_forcing_ratio
           
           # 확률 가장 높게 예측한 토큰
            top1 = output.argmax(1) 
           
           # techer_force = 1 = True이면 trg[t]를 아니면 top1을 input으로 사용
            input = trg[i] if teacher_force else top1
       
        return outputs
       