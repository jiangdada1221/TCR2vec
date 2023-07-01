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

def rename_Vseg(Vname):
    if Vname[1] == 'C':
        Vname = 'TRB' + Vname[4:]
        if Vname[Vname.find('-') + 1] == '0':
            Vname = Vname[:(Vname.find('-') +
                            1)] + Vname[(Vname.find('-') + 2):]
        if Vname[Vname.find('V') + 1] == '0':
            Vname = Vname[:(Vname.find('V') +
                            1)] + Vname[(Vname.find('V') + 2):]
    return Vname


def rename_Jseg(Jname):
    if Jname[1] == 'C':
        Jname = 'TRB' + Jname[4:]

        if Jname[Jname.find('-') + 1] == '0':
            Jname = Jname[:(Jname.find('-') +
                            1)] + Jname[(Jname.find('-') + 2):]
        if Jname[Jname.find('J') + 1] == '0':
            Jname = Jname[:(Jname.find('J') +
                            1)] + Jname[(Jname.find('J') + 2):]
    return Jname

def map_gene(gene_name,ref_gene):
    #Hierachy mapping TRBVx/TRBJx
    ref_gene = ref_gene.split('|')[1]
    if gene_name == ref_gene :
        return True
    if str.isnumeric(gene_name[5:6]):
        gene_family = gene_name[4:6]
    else:
        gene_family = gene_name[4:5]
    if str.isnumeric(ref_gene[5:6]):
        ref_family = ref_gene[4:6]
    else:
        ref_family = ref_gene[4:5]
    if gene_family != ref_family:      
        return False
    #family mapped
    # print(gene_name + '  ' + ref_gene)
    if gene_name in ref_gene:
        return True
    if len(gene_name.split('-')) > 1:
       gene_split = gene_name.split('-')[1] 
       gene_f2 = gene_split[0]
       if len(ref_gene.split('-')) == 1:
           return False #ref has no allele
       if ref_gene.split('-')[1][0] != gene_f2:
           return False
    # if family level
    return True
    

def to_full_seq(directory, Vname, Jname, CDR3):
    ## Translate segment name into segment sequence
    foundV = False
    foundJ = False
    Vseq = ''
    Jseq = ''
    # for i in range(1,9):
    #     if f'-0{i}' in Vname:
    #         Vname = Vname.replace(f'-0{i}',f'-{i}')
    #         break
    # if ':' in Jname:
    #     Jname = Jname.replace(':','-')    
    for Vrecord in SeqIO.parse(
        os.path.join(directory, 'V_segment_sequences.fasta'), "fasta"
    ):
        if type(Vname) != str or Vname == 'unresolved':
            print('Vname not string but ', Vname, type(Vname))
            Vseq = ''

        else:
            ## Deal with inconsistent naming conventions of segments
            Vname_adapted = rename_Vseg(Vname)     
            if not 'TRBV' in Vrecord.id:                
                continue
            # if Vname_adapted in Vrecord.id:            
            if map_gene(Vname_adapted,Vrecord.id):
                Vseq = Vrecord.seq
                foundV = True                        
            # elif '-' in Vname_adapted:                
            #     Vname_adapted = Vname_adapted.split('-')[0]                
            #     if Vname_adapted in Vrecord.id:
            #         Vseq = Vrecord.seq
            #         foundV = True                    
        if foundV:
            break   
    # print(foundV)             
    for Jrecord in SeqIO.parse(
        os.path.join(directory, 'J_segment_sequences.fasta'), "fasta"
    ):
        if type(Jname) != str or Jname == 'unresolved':
            print('Jname not string but ', Jname, type(Jname))
            Jseq = ''
        else:
            ## Deal with inconsistent naming conventions of segments
            Jname_adapted = rename_Jseg(Jname)
            if not 'TRBJ' in Jrecord.id:
                continue
            # if Jname_adapted in Jrecord.id:
            if map_gene(Jname_adapted,Jrecord.id):
                Jseq = Jrecord.seq
                foundJ = True
        if foundJ:
            break
    if foundV and Vseq != '':
        ## Align end of V segment to CDR3
        alignment = pairwise2.align.globalxx(
            Vseq[-5:],  # last five amino acids overlap with CDR3
            CDR3,
            one_alignment_only=True,
            penalize_end_gaps=(False, False)
        )[0]
        best = list(alignment[1])

        ## Deal with deletions
        if best[0] == '-' and best[1] == '-':
            best[0] = Vseq[-5]
            best[1] = Vseq[-4]
        if best[0] == '-':
            best[0] = Vseq[-5]

        # remove all left over -
        best = "".join(list(filter(lambda a: a != '-', best)))
    else:
        best = CDR3

    ## Align CDR3 sequence to start of J segment
    if Jseq != '':
        alignment = pairwise2.align.globalxx(
            best,
            Jseq,
            one_alignment_only=True,
            penalize_end_gaps=(False, False)
        )[0]

        # From last position, replace - with J segment amino acid
        # until first amino acid of CDR3 sequence is reached
        best = list(alignment[0])[::-1]
        firstletter = 0
        for i, aa in enumerate(best):
            if aa == '-' and firstletter == 0:
                best[i] = list(alignment[1])[::-1][i]
            else:
                firstletter = 1

        # remove all left over -
        best = "".join(list(filter(lambda a: a != '-', best[::-1])))

    full_sequence = Vseq[:-5] + best

    return full_sequence, foundV, foundJ

def divide_chunks(samples, n):     
    # looping till length l
    for i in range(0, len(samples), n):
        yield samples[i:i + n]

def cdr2full(directory,samples,verbose=True,multi_process=False):
    if multi_process:
        if len(samples) > 5e6:
            print('using batched')
            sample_ref = samples
            batch_num = len(samples) // 1000000 + 1 if len(samples) % 1000000 != 0 else len(samples) // 1000000
            full_seqs = []
            for i in range(batch_num):
                print('total batch: ',batch_num,'current batch: ',i+1)
                samples = sample_ref[i * 1000000:(i+1)*1000000]
                processes = mp.cpu_count() 
                print(len(samples))
                n = ceil(len(samples) / processes)                  
                samples_chunks = list(divide_chunks(samples, n))
                args = [(directory,s) for s in samples_chunks]
                pool = mp.Pool(processes=processes)        
                full_seq = pool.map(_wrapper,args)
                full_seq = np.concatenate(full_seq)
                pool.close()
                pool.join()             
                full_seqs.append(full_seq)
            full_seqs = np.concatenate(full_seqs)
        else :     
            processes = mp.cpu_count() 
            print(len(samples))
            n = ceil(len(samples) / processes)                  
            samples_chunks = list(divide_chunks(samples, n))
            args = [(directory,s) for s in samples_chunks]
            pool = mp.Pool(processes=processes)        
            full_seqs = pool.map(_wrapper,args)
            full_seqs = np.concatenate(full_seqs)
            pool.close()
            pool.join()                        
        # for s in full_seqs:
        #     assert 'Failure' != s
    else :
        full_seqs = _cdr2full(directory,samples,verbose)
    return full_seqs

def _wrapper(args):
   return _cdr2full(*args)

def _cdr2full(directory,samples,verbose=False):
    full_seq = []
    if not verbose:
        tqdm = list
    else :
        from tqdm import tqdm
    for sample in tqdm(samples):
        full,foundV,_ = to_full_seq(directory,str(sample[1]), str(sample[2]), str(sample[0]))        
        if foundV:
            full_seq.append(full._data)        
        else :
            full_seq.append('Failure')         
    return full_seq

