import torch
import torch.nn as nn
import sys
from tape import ProteinBertModel, TAPETokenizer
import torch.nn.functional as F
from tcr2vec.Epi_model.epi_encode_ed import tcrgp_encoding

class TCR2vec(nn.Module):
    def __init__(self,path):
        '''
        @path: the path to the pretrained transformer folder
        '''
        super().__init__()
        self.model = ProteinBertModel.from_pretrained(path)
        self.model.eval() #default is the eval mode

    def forward(self,batch,detach=True):
        if type(batch) != dict: #for batch input
            output = self.model(batch)
            output = output[0].mean(dim=1) #when only 1 input, no need to filter out [pad] tokens
        else :
            output = self.emb(batch,detach)    
        
        return output

    def emb(self,batch,detach=True):
        output = self.model(batch['input_ids'],input_mask=batch['input_mask'])                  
        output = output[0]
        # print(output[0].shape)                  
        mask = batch['input_mask']
        bs,L = mask.shape
        emb_size = output.shape[-1]
        masks = mask.repeat(emb_size,1,1)        
        masks = masks.permute((1,2,0))        
        output = output * masks
        output = torch.sum(output,1)                
        output = output / torch.sum(mask,1).view(-1,1)            
        if not detach:
            return output
        else :
            return output.detach().cpu().numpy()

class MLP(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim=1,dropout=0.1):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hid_dim), 
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hid_dim, hid_dim // 4), 
            nn.BatchNorm1d(hid_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hid_dim // 4, out_dim))

    def forward(self, x): 
        x = self.project(x)       
        return x

class Finetune_head(nn.Module):
    def __init__(self,model,feature_size,hidden_size,dropout=0.0,device='cuda:0'):
        super().__init__()
        self.model = model
        self.proj = MLP(feature_size,hidden_size,dropout=dropout)
        self.sigmoid = nn.Sigmoid()
        self.device=device

    def forward(self,batch):

        emb = self.model(batch,detach=False)
        emb_epitope = torch.FloatTensor(tcrgp_encoding(batch['epitope'],MAXLEN=15,d=8)).to(self.device)
        emb = torch.cat((emb,emb_epitope),1)        
        emb = self.proj(emb)
        emb = self.sigmoid(emb)
        return emb

