import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

def pdist(vectors): 
    #compute the pairwise eulidean distances (squared) for the given embedding matrix (N x d)
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    distance_matrix = distance_matrix.fill_diagonal_(0)
    return distance_matrix

def pdist_np(vectors):
    distance_matrix = -2 * vectors@(np.transpose(vectors)) + (vectors**2).sum(axis=1).reshape(1, -1) + (vectors**2).sum(
        axis=1).reshape(-1, 1)
    np.fill_diagonal(distance_matrix,0)
    return distance_matrix

def correlation(X,Y):
    #linear correlation between two vectors X and Y
    return np.sum((X -np.mean(X)) * (Y - np.mean(Y)) ) / (np.sqrt(np.sum((X-np.mean(X))**2) * np.sum((Y-np.mean(Y))**2)))

def get_emb(tcr2vec,loader,detach=True):
    '''
    Get the embeddings from TCRvec model
    @tcr2vec: model
    @loader: the loader 
    @detach: if True, will detach from the computation graph. i.e. you will get numpy array; 
             if False, will return the Tensor object;
    '''
    emb = []
    tcr2vec.eval()
    device = next(tcr2vec.parameters()).device
    with torch.no_grad():
        for batch in tqdm(loader):
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['input_mask'] = batch['input_mask'].to(device)
            #self.model.l2_normalized=False
            emb_b = tcr2vec(batch,detach=detach)                
            #emb_b = emb_b.detach().cpu().numpy() #b x emb
            emb.append(emb_b) 
    if detach:                               
        emb = np.concatenate(emb)
    else :
        emb = torch.cat(emb,0)
    return emb

