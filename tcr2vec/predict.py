import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from tape import ProteinBertModel
from tcr2vec.dataset import TCRLabeledDset
from tcr2vec.model import TCR2vec,Finetune_head
from tqdm import tqdm
import torch.optim as optim
import os
from tcr2vec.utils import get_emb
import torch.nn as nn
import argparse
from sklearn.metrics import roc_auc_score as AUC
from tcr2vec.Epi_model.epi_encode_ed import tcrgp_encoding
import pickle
import gc

# Predict the 
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dset_path',type=str)
    parser.add_argument('--batch_size',type=int,default=128)    
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--save_prediction_path',type=str)
    parser.add_argument('--model_path',type=str,help='The path to the saved prediction model. (xx.pickle or xx.pth)')
    parser.add_argument('--use_column',type=str,default='full_seq')
    parser.add_argument('--pretrain_path',type=str,default='None',help='The pretrained TCR2vec model; only used when the "model_path" is xx.pickle')
    parser.add_argument('--epi_encode',type=str,default='eigendecom',help='The method for encoding epitopes; either eigendecom or atchleyae; only used when the "model_path" is xx.pickle')
    args = parser.parse_args()

    file_path = args.dset_path

    finetune = True
    if args.model_path.endswith('.pickle'):
        finetune=False

    if finetune:
        model_ft = torch.load(args.model_path,map_location=args.device) #direct load model; 
        model_ft.eval()


    dset = TCRLabeledDset(file_path,only_tcr=False,use_column=args.use_column)
    loader = DataLoader(dset,batch_size=args.batch_size,collate_fn=dset.collate_fn,shuffle=False) #remember to keep shuffle=False!!!
    trues,pres = [],[]

    if finetune:
        with torch.no_grad():
            for batch in tqdm(loader):
                    keys = ['input_ids','input_mask']
                    for k in keys:
                        batch[k] = batch[k].to(args.device) 
                    out_logits = model_ft(batch) #b x 1
                    out_logits = out_logits[:,0].detach().cpu().numpy()
                    labels = batch['label']
                    trues.extend(labels)
                    pres.extend(out_logits)
    else :
        clf = pickle.load(open(args.model_path,'rb'))
        tcr2vec = TCR2vec(args.pretrain_path).to(args.device)
        tcr2vec.eval()
        epitopes = pd.read_csv(file_path)['Epitope'].values
        trues = pd.read_csv(file_path)['Label'].values

        if args.epi_encode == 'atchleyae':        
            from tensorflow import keras
            from tensorflow.keras import backend as K
            from tensorflow.keras.models import load_model, Model
            from tensorflow.compat.v1 import ConfigProto
            from tensorflow.compat.v1 import InteractiveSession
            from tcr2vec.Epi_model.epi_encode_ae import DataGenerator
            TESSA =load_model('Epi_model/ept_trained_AE.h5') 
            epi_model = Model(TESSA.input,TESSA.layers[-12].output)
            import csv
            aa_dict=dict()         # read Atchley factors for mapping every amino acid to corresponding vectors
            print('Using AtchleyAE to encode epitopes')
            with open('Epi_model/Atchley_factors.csv','r') as aa:
                aa_reader=csv.reader(aa)
                next(aa_reader, None)
                for rows in aa_reader:
                    aa_name=rows[0]
                    aa_factor=rows[1:len(rows)]
                    aa_dict[aa_name]=np.asarray(aa_factor,dtype='float')
            dset = DataGenerator(epitopes, aa_dict, encode_dim=80, batch_size=32, shuffle=False)
            emb_epi = []
            for b in dset:
                if type(b) == int:
                    continue
                emb_epi.append(epi_model(b).numpy())
            emb_epi = np.concatenate(emb_epi,0)

            del TESSA,epi_model
            gc.collect()

        else :
            emb_epi = tcrgp_encoding(epitopes,MAXLEN=15,d=8)

        emb_tcr = get_emb(tcr2vec,loader)
        
        emb_epi = emb_epi/np.linalg.norm(emb_epi,axis=1).reshape((len(emb_epi),1))
        emb_tcr = emb_tcr/np.linalg.norm(emb_tcr,axis=1).reshape((len(emb_tcr),1))

        emb = np.concatenate((emb_tcr,emb_epi),1)
        pres = clf.predict_proba(emb)[:,1]

    print(AUC(trues,pres))
    print(f'Record the prediction results to {args.save_prediction_path}')
    with open(args.save_prediction_path,'w') as f:
        for i in range(len(trues)):
            f.write(str(pres[i]) + ',' + str(trues[i])+'\n')