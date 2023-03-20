from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from tape import TAPETokenizer
from collections import defaultdict

def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class TCRLabeledDset(Dataset):    
    def __init__(self,file,only_tcr=False,use_column = 'full_seq'):
        '''
        The dataset module for TCR data. 
        @file: the path to the TCR dataset (.csv file). 
        @only_tcr: if set to True, will only load the TCR data (specified by the "use_column"); 
                   if set to False, will load TCR data; Epitope data; corresponding Labels
        @use_column: The column name for TCR data; By default, the column name for epitopes is "Epitope"; the name for labels is "Label" 
        Note that, you can also input a list of TCR (file=TCR_list)
        '''
        if type(file) == str:
            d = pd.read_csv(file)
        else :
            d = pd.DataFrame({use_column:file})
        self.only_tcr = only_tcr
        if only_tcr:
            cs = d[use_column].values
            self.seqs = cs
        else: 
            cs,es = d[use_column].values, d['Epitope'].values #### #original is 'Label'
            self.seqs = cs
            self.labels = d['Label'].values
            self.epitopes = es                

        self.tokenizer = TAPETokenizer(vocab='iupac')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self,idx):
        seq = self.seqs[idx]        
        tokens = self.tokenizer.tokenize(seq)
        tokens = self.tokenizer.add_special_tokens(tokens)
        token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(tokens), np.int64)                
        input_mask = np.ones_like(token_ids)
        if not self.only_tcr:
            label = self.labels[idx]
            epitope = self.epitopes[idx]
            #label = self.e2l[label]
            return token_ids,input_mask,label,epitope, seq
        else :
            return token_ids, input_mask,seq

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        # input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))
        if not self.only_tcr:
            input_ids, input_mask, label,epitope, seq = tuple(zip(*batch))
        else :
            input_ids, input_mask, seq = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        #label = torch.from_numpy(np.array(list(label)))
        # ignore_index is -1 
        if not self.only_tcr:
            return {'input_ids': input_ids,
                'input_mask': input_mask,
                'label':list(label),
                'seq':list(seq),
                'epitope':list(epitope)}
        else :
            return {'input_ids': input_ids,
                'input_mask': input_mask,
                'seq':list(seq)}
