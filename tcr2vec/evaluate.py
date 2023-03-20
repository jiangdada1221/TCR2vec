import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import gc
from dataset import TCRLabeledDset
from torch.utils.data import DataLoader
import sklearn
from sklearn import svm
from tcr2vec.utils import get_emb
from tcr2vec.Epi_model.epi_encode_ed import tcrgp_encoding
import argparse
from sklearn.neural_network import MLPClassifier as MLP
import os
from sklearn.metrics import roc_auc_score as AUC
from tcr2vec.model import TCR2vec
import pickle

'''
Classification performance using SVM/MLP classification head. 
'''

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dset_folder',type=str,help='The path to the 5fold dataset; column names should contain CDR3.beta/full_seq;Epitope;Label')
    parser.add_argument('--batch_size',type=int,default=128)  
    parser.add_argument('--device',type=str,default='cuda:0',help='device; ''cpu'' or cuda:x')  
    parser.add_argument('--pretrain_path',type=str,help='The path to the pretrained TCRvec model')
    parser.add_argument('--use_column',type=str,default='full_seq',help='The column in dset_folder to be used; either CDR3.beta or full_seq')
    parser.add_argument('--record_path',type=str,default='None',help='The path to record the 5fold prediction scores;')  
    parser.add_argument('--c_method',type=str,default='SVM',help='Classification method; "SVM" or "MLP"') 
    parser.add_argument('--epi_encode',type=str,default='eigendecom',help='The method to encode epitope sequences; "eigendecom" or "atchleyae"')            
    parser.add_argument('--save_path',type=str,default='None',help='The path to save the prediction model (.pickle)')
    parser.add_argument('--use_sklearnx',type=str,default='False',help='Whether to use the sklearnx package for accelerating the sklearn models') 
    args = parser.parse_args()
    if args.use_sklearnx == 'True':
        from sklearnex import patch_sklearn
        patch_sklearn()

    model_path = args.pretrain_path
    folder = args.dset_folder
    model = TCR2vec(model_path).to(args.device)
    model.eval()

    for fold in range(1): #5-fold cross-validation procedure
        print('The current fold is :', fold)
        train = folder + f'{fold}_train.csv'        
        test = folder + f'{fold}_test.csv'
        train_dset = TCRLabeledDset(train,only_tcr=False,use_column = args.use_column)
        test_dset = TCRLabeledDset(test,only_tcr=False,use_column = args.use_column) # for e2l #!!!!!! 

        epitope_train,epitope_test = pd.read_csv(train)['Epitope'].values, pd.read_csv(test)['Epitope'].values
        #encode epitope sequences
        if args.epi_encode == 'eigendecom':
            epi_train_emb = tcrgp_encoding(epitope_train,MAXLEN=15,d=8)
            epi_test_emb = tcrgp_encoding(epitope_test,MAXLEN=15,d=8)
            print('Using Eigdecom to encode epitopes')
        elif args.epi_encode == 'atchleyae':        
            from tensorflow import keras
            from tensorflow.keras import backend as K
            from tensorflow.keras.models import load_model, Model
            from tensorflow.compat.v1 import ConfigProto
            from tensorflow.compat.v1 import InteractiveSession
            from tcr2vec.Epi_model.epi_encode_ae import DataGenerator
            TESSA =load_model('../tcr2vec/Epi_model/ept_trained_AE.h5') 
            epi_model = Model(TESSA.input,TESSA.layers[-12].output)
            import csv
            aa_dict=dict()         # read Atchley factors for mapping every amino acid to corresponding vectors
            print('Using AtchleyAE to encode epitopes')
            with open('../tcr2vec/Epi_model/Atchley_factors.csv','r') as aa:
                aa_reader=csv.reader(aa)
                next(aa_reader, None)
                for rows in aa_reader:
                    aa_name=rows[0]
                    aa_factor=rows[1:len(rows)]
                    aa_dict[aa_name]=np.asarray(aa_factor,dtype='float')
            dset_train = DataGenerator(epitope_train, aa_dict, encode_dim=80, batch_size=32, shuffle=False)
            dset_test = DataGenerator(epitope_test, aa_dict, encode_dim=80, batch_size=32, shuffle=False)
            emb_train,emb_test = [],[]
            for b in dset_train:
                if type(b) == int:
                    continue
                emb_train.append(epi_model(b).numpy())
            for b in dset_test:
                if type(b) == int:
                    continue
                emb_test.append(epi_model(b).numpy())
            epi_train_emb = np.concatenate(emb_train,0)
            epi_test_emb = np.concatenate(emb_test,0)
            del TESSA,epi_model
            gc.collect()

        #Normalization
        epi_train_emb = epi_train_emb/np.linalg.norm(epi_train_emb,axis=1).reshape((len(epi_train_emb),1))
        epi_test_emb = epi_test_emb/np.linalg.norm(epi_test_emb,axis=1).reshape((len(epi_test_emb),1))

        label_train = train_dset.labels 
        label_test = test_dset.labels

        loader_train = DataLoader(train_dset,batch_size=args.batch_size,collate_fn=train_dset.collate_fn,shuffle=False) 
        loader_test = DataLoader(test_dset,batch_size=args.batch_size,collate_fn=test_dset.collate_fn,shuffle=False)

        emb_train = get_emb(model,loader_train)            
        emb_test = get_emb(model,loader_test) 

        #Normalization
        emb_train =  emb_train/np.linalg.norm(emb_train,axis=1).reshape((len(emb_train),1))
        emb_test = emb_test/np.linalg.norm(emb_test,axis=1).reshape((len(emb_test),1))

        #concatenate tcr and epitope embedding
        emb_train = np.concatenate((emb_train,epi_train_emb),1)
        emb_test = np.concatenate((emb_test,epi_test_emb),1)
        
        print(f'Begin training the {args.c_method}')
        from sklearn.linear_model import SGDClassifier as SGD
        if args.c_method == 'SVM':
            accs,aucs = [],[]                        
            accs,aucs = [],[]                        
            clf = svm.SVC(C=  10,probability=True)                    
            clf.fit(emb_train,label_train)
            predicted = clf.predict_proba(emb_test)
            #predicted = clf.decision_function(emb_test)                    
            predicted = predicted[:,1]
            #predicted_class = clf.predict(emb_test)
            print('AUC: ', AUC(label_test,predicted))            
            if args.record_path != 'None':
                print(f'Record the classification results to {args.record_path}_{fold}_scores.txt')
                with open(args.record_path + f'{fold}_scores.txt','w') as f:
                    for i in range(len(predicted)):
                        f.write(str(predicted[i]) + ',' + str(label_test[i])+'\n')
            if args.save_path != 'None':
                print(f'Save the classification model to {args.save_path}_{fold}.pickle')
                pickle.dump(clf,open(args.save_path + f'_{fold}.pickle','wb'))     

        elif args.c_method == 'MLP':
            #original is 50 max_iter
            clf = MLP(hidden_layer_sizes=(256,128,),max_iter=100,validation_fraction=0.2,solver='adam',batch_size=128,early_stopping=True).fit(emb_train,label_train)
            predicted = clf.predict_proba(emb_test)
            predicted_class = clf.predict(emb_test)
            print(f'The AUC: ', AUC(label_test,predicted[:,1]))
            if args.record_path != 'None':
                print(f'Record the classification results to {args.record_path}_{fold}_scores.txt')
                with open(args.record_path + f'{fold}_scores.txt','w') as f:
                    for i in range(len(predicted)):
                        f.write(str(predicted[i][1]) + ',' + str(label_test[i])+'\n')
            if args.save_path != 'None':
                print(f'Save the classification model to {args.save_path}_{fold}.pickle')
                pickle.dump(clf,open(args.save_path + f'_{fold}.pickle','wb'))

# ! The clustering code is not provided now since it depends on the GIANA package and clusTCR package
# TCRdist: https://github.com/svalkiers/clusTCR/tree/main/clustcr/modules/tcrdist
# GIANA: https://github.com/s175573/GIANA 
