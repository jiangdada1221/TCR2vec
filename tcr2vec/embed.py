import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb
from tcr2vec.model import TCR2vec
import argparse

#embed TCRs for a given file (in csv format); 
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dset_path',type=str,help='Path to the file recording the TCRs that users want to embed')
    parser.add_argument('--column_name',type=str,default='full_seq',help='Column name of the provided file recording TCRs') 
    parser.add_argument('--pretrain_path',type=str,help='The path to the pretrained TCR2vec model')
    parser.add_argument('--batch_size',type=int,default=128) 
    parser.add_argument('--device',type=str,default='cuda:0')   
    parser.add_argument('--save_path',type=str,default='embedding.npy', help='The path to store the embeddings. Currently it should be in .npy format')
    args = parser.parse_args()

    model_path = args.pretrain_path
    model = TCR2vec(model_path).to(args.device)
    model.eval()

    dset = TCRLabeledDset(args.dset_path,only_tcr=True,use_column=args.column_name)
    loader = DataLoader(dset,batch_size=args.batch_size,collate_fn=dset.collate_fn,shuffle=False) 

    emb = get_emb(model,loader) 
    print(f'Saving the embedding to {args.save_path}')
    np.save(args.save_path,emb)
