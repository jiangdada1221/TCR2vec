from tcr2vec.model import TCR2vec
from tape import TAPETokenizer
import torch
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb

if __name__ == '__main__':
    path_to_TCR2vec = '../tcr2vec/models/TCR2vec_120'
    emb_model = TCR2vec(path_to_TCR2vec)
    tokenizer = TAPETokenizer(vocab='iupac') 

    #example TCR
    seq = 'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLRLIYYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV'
    token_ids = torch.tensor([tokenizer.encode(seq)])
    output = emb_model(token_ids) # shape of 1 x 128

    #convert to numpy array
    emb = output.detach().cpu().numpy()    

    #for batch input:
    dset = TCRLabeledDset([seq],only_tcr=True)
    loader = DataLoader(dset,batch_size=32,collate_fn=dset.collate_fn,shuffle=False)
    emb = get_emb(emb_model,loader,detach=True)