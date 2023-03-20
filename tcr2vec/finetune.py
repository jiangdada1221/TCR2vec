import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from tcr2vec.dataset import TCRLabeledDset
from tcr2vec.model import TCR2vec,Finetune_head
from tqdm import tqdm
import torch.optim as optim
import os
import torch.nn as nn
from tcr2vec.utils import get_emb
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import roc_auc_score as AUC

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_train',type=str,help='The path to the training file')
    parser.add_argument('--path_test',type=str,default='None',help='The path to the test file')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=20)
    parser.add_argument('--emb_size',type=int,default=120,help='Embedding size of TCR')
    parser.add_argument('--emb_size_epi',type=int,default=120,help='Embedding size of epitope; 120 for eigdecom; 128 for atchleyae')
    parser.add_argument('--save_per_epoch',type=int,default=2,help='will save the model after each "save_per_epoch" epoch')
    parser.add_argument('--device',type=str,default='cuda:0')    
    parser.add_argument('--save_path',type=str,default='None',help='The path to save the model. should be ends with .pth')
    parser.add_argument('--save_prediction_path',type=str,default='None',help='The path to save the predictions evaluted on test set; only useful when path_test != None')
    parser.add_argument('--pretrain_path',type=str)
    parser.add_argument('--use_column',type=str,default='full_seq')
    parser.add_argument('--learning_rate',type=float,default=1e-4)    
    parser.add_argument('--dropout',type=float,default=0.0) 
    args = parser.parse_args()

    loss_fcn = torch.nn.BCELoss()
    print('Begin finetuning')
    
    tcr2vec = TCR2vec(args.pretrain_path).to(args.device)
    tcr2vec.train() #fine tune mode

    model_ft = Finetune_head(tcr2vec,args.emb_size_epi+args.emb_size,128,dropout=args.dropout).to(args.device)
    # total_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    # print('The total number of trainable parameters: ',total_params)

    train = args.path_train
    test = args.path_test
    evaluate = True if test != 'None' else False

    train_dset = TCRLabeledDset(train,only_tcr=False,use_column=args.use_column)
    loader_train = DataLoader(train_dset,batch_size=args.batch_size,collate_fn=train_dset.collate_fn,shuffle=True) 

    if evaluate:
        test_dset = TCRLabeledDset(test,only_tcr=False,use_column=args.use_column) 
        loader_test = DataLoader(test_dset,batch_size=args.batch_size,collate_fn=test_dset.collate_fn,shuffle=False)        
    
    optimizer = optim.Adam(model_ft.parameters(), lr=args.learning_rate)
    scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)        
    epochs = args.epoch    
    #training 
    for e in range(epochs) : 
        model_ft.train()
        loss_in_epoch = []
        for batch in tqdm(loader_train):
            keys = ['input_ids','input_mask']
            for k in keys:
                batch[k] = batch[k].to(args.device) 
            out_logits = model_ft(batch) #b x 1
            labels = torch.FloatTensor(batch['label']).to(args.device)
            loss = loss_fcn(out_logits[:,0],labels)
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
            scheduler.step()        
            loss_in_epoch.append(loss.item())            

        print(f'Average training loss for {e+1} epoch : ', np.mean(loss_in_epoch))        
        if evaluate:
            model_ft.eval()
            print('Begin evaluating')
            pres,trues = [],[]
            with torch.no_grad():
                for batch in tqdm(loader_test):
                    keys = ['input_ids','input_mask']
                    for k in keys:
                        batch[k] = batch[k].to(args.device)      
                    out_logits = model_ft(batch).detach().cpu().numpy()
                    #out_logits = model_ft(list(batch['seq'])).detach().cpu().numpy()
                    labels = list(batch['label'])
                    trues.extend(labels)                
                    pres.extend(out_logits[:,0])
            print('The AUC for the test: ',AUC(trues,pres))            
            
        if e % args.save_per_epoch and args.save_path != 'None':
            save_path = args.save_path
            print('Saving the model to {}'.format(save_path))
            torch.save(model_ft,save_path)
            
    if evaluate:
        print(f'Saving the predictions to {args.save_prediction_path}')
        with open(args.save_prediction_path,'w') as f:
            for i in range(len(trues)):
                f.write(str(pres[i]) + ',' + str(trues[i])+'\n')
    if args.save_path != 'None':
        print('Saving the final model to {}'.format(args.save_path))
        torch.save(model_ft,args.save_path) ###need to check that

            






