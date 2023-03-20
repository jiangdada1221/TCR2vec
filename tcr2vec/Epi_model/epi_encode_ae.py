#for ae-atchley
from torch.utils.data import Dataset,DataLoader
from tensorflow import keras
import numpy as np

def aamapping(peptideSeq,aa_dict,encode_dim):
    #Transform aa seqs to Atchley's factors.
    peptideArray = []
    if len(peptideSeq)>encode_dim:
        print('Length: '+str(len(peptideSeq))+' over bound!')
        peptideSeq=peptideSeq[0:encode_dim]
    for aa_single in peptideSeq:
        try:
            peptideArray.append(aa_dict[aa_single])
        except KeyError:
            print('Not proper aaSeqs: '+peptideSeq)
            peptideArray.append(np.zeros(5,dtype='float64'))
    for i in range(0,encode_dim-len(peptideSeq)):
        peptideArray.append(np.zeros(5,dtype='float64'))
    return np.asarray(peptideArray)
    
def tcr2vec(dataset,aa_dict,encode_dim):
    #Wrapper of aamapping
    vecs = []
    for tcr in dataset:
        vecs.append(aamapping(tcr,aa_dict,encode_dim))

    return vecs

class DataGenerator(keras.utils.Sequence):
    def __init__(self, datapath, aa_dict, encode_dim = 30, subset="training", batch_size=32, shuffle=True):
        if type(datapath)!= str:
            # datapath = pd.DataFrame({'CDR3.beta':datapath})
            self.TCRseqs = list(datapath)
        else :
            self.TCRseqs = pd.read_csv(datapath)["CDR3.beta"].to_list()
        # if subset=="training":
        #     self.dataset = np.array(self.TCRseqs[:int(len(self.TCRseqs)*0.9)])
        # else:
        #     self.dataset = np.array(self.TCRseqs[int(len(self.TCRseqs)*0.9):])
        self.dataset = np.array(self.TCRseqs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.encode_dim = encode_dim
        self.aa_dict = aa_dict
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle  == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        end = self.batch_size * (index+1) if self.batch_size * (index+1) < self.__len__() else self.__len__()
        indexes = self.indexes[index*self.batch_size:end]
        if len(indexes) == 0:
            return -1
        TCR_vecs=tcr2vec(self.dataset[indexes], self.aa_dict, self.encode_dim)
        TCR_vecs=np.stack(TCR_vecs)
        TCR_vecs=TCR_vecs.reshape(-1,self.encode_dim,5,1)

        return TCR_vecs

    def __len__(self):

        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.dataset) / self.batch_size))
        return len(self.TCRseqs)