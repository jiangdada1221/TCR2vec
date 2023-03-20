import numpy as np
from sklearn.decomposition import PCA as PCA
import tcr2vec
#EigenDecom method for embedding epitopes (TCRs)

alphabet='ARNDCEQGHILKMFPSTWYV-'
# MAXLEN = 30
idx2aa = {i:alphabet[i] for i in range(len(alphabet))}

def tcrs2nums(tcrs):
    """Converts a list of (TCR) amino acid sequences to numbers. Each letter is changed to its index in the alphabet"""
    tcrs_num=[]
    n=len(tcrs)
    for i in range(n):
        t=tcrs[i]
        nums=[]
        for j in range(len(t)):
            nums.append(alphabet.index(t[j]))
        tcrs_num.append(nums)
    return tcrs_num

def add_gap(tcr,l_max,gap_char='-'):
    """Add gap to given TCR. Returned tcr will have length l_max.
    If there is an odd number of letters in the sequence, one more letter is placed in the beginning."""  
    l = len(tcr)
    if l<l_max:
        i_gap=np.int32(np.ceil(l/2))
        tcr = tcr[0:i_gap] + (gap_char*(l_max-l))+tcr[i_gap:l]
    return tcr

def check_align_cdr3s(cdr3s,lmaxtrain = 30):
    """Check cdr3s for too long sequences or sequences containing characters outside alphabet
    returns cdr3s_letter (proper cdr3s aligned, but improper sequences are left as they are)
            cdr3s_aligned (proper cdr3s aligned, places of improper sequences are left empty),
            and Ikeep3 (locations of proper cdr3s)
    Here improper means sequences that are longer than those in the training data or contain
    characters outside the used alphabet."""
    lmaxtest = lmaxtrain
    Ikeep3=np.ones((len(cdr3s),),dtype=bool)
    cdr3s_aligned=[]
    cdr3s_letter =[]

    for i in range(len(cdr3s)):
        ca = add_gap(cdr3s[i],lmaxtrain)
        cdr3s_aligned.append(ca)
        cdr3s_letter.append(ca)
    return cdr3s_letter, cdr3s_aligned, Ikeep3

def seq2numlist(cdr3as, balance_controls=False,MAXLEN=30):  ## AA -> [20, 3, 14 ... ]
    Itest = np.ones(len(cdr3as),dtype=bool)
    seq_lists = []
    cdr3as_letter,cdr3as, I = check_align_cdr3s(cdr3as,MAXLEN) # I: which cdr3s will be kept
    if balance_controls:
        Itest[np.logical_and( Ie,~I)]=False
        Itest[~Ie] = Itest[Ie]
    else:
        Itest[~I]=False
    # if clip3:  ## clip: list, remove clip[0] amino acids from beginning and clip[1] amino acids from the end
    #     cdr3as = clip_cdr3s(cdr3as,clip)
    seq_lists.append(tcrs2nums(cdr3as))
    return seq_lists


def subsmatFromAA2(identifier,data_file='../tcr2vec/Epi_model/aaindex2.txt'):
    """Retrieve a substitution matrix from AAindex2-file, scale it between 0 and 1, and add gap"""
    with open(data_file,'r') as f:
        for line in f:
            if identifier in line:
                break
        for line in f:            
            if line[0] == 'M':
                split_line=line.replace(',',' ').split()
                rows=split_line[3]
                cols=split_line[6]
                break

        subsmat=np.zeros((21,21),dtype=np.float)
        i0=0
        for line in f:
            i=alphabet.find(rows[i0])
            vals=line.split()   
            for j0 in range(len(vals)):
                j=alphabet.find(cols[j0])
                subsmat[i,j]=vals[j0]
                subsmat[j,i]=vals[j0]
            i0+=1    
            if i0>=len(rows):
                break        
    subsmat[:-1,:-1]+=np.abs(np.min(subsmat))+1
    subsmat[:-1,:-1]/=np.max(subsmat)
    subsmat[-1,-1]=np.min(np.diag(subsmat)[:-1])
    
    return subsmat    

def get_pcs(subsmat,d):
    """Get first d pca-components from the given substitution matrix."""
    pca = PCA(d)
    pca.fit(subsmat)
    pc = pca.components_
    return pc


def encode_with_pc(seq_lists, lmaxes, pc):
    """ Encode the sequence lists (given as numbers), with the given pc components (or other features)
    lmaxes contains the maximum lengths of the given sequences. """
    d = pc.shape[0]
    # X = np.zeros((len(seq_lists[0]),d*sum(lmaxes)))
    X = np.zeros((len(seq_lists[0]),d*sum(lmaxes)))
    i_start, i_end = 0, 0
    for i in range(len(seq_lists)):
        Di = d*lmaxes[i]
        i_end += Di
        for j in range(len(seq_lists[i])):
            X[j,i_start:i_end] = np.transpose( np.reshape( np.transpose( pc[:,seq_lists[i][j]] ), (Di,1) ) )
        i_start=i_end
    return X

# The EigenDecom encoding
def tcrgp_encoding(seq_lists,subsmat_code = 'HENS920102',MAXLEN=30,d=4):
    cdr_lists = seq2numlist(seq_lists,MAXLEN=MAXLEN)
    subsmat = subsmatFromAA2(subsmat_code)
    pc_blo = get_pcs(subsmat,d=d)
    X = encode_with_pc(cdr_lists,lmaxes = [MAXLEN], pc = pc_blo)
    return X
