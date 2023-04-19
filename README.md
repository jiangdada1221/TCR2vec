# TCR2vec
TCR2vec is a python software designed for embedding TCR sequences into numerical vectors. It is a transformer-based model that pretrained with __MLM__ and __SPM__ (similarity preservation modeling). After the multi-task pretraining stage, TCR2vec is able to transform amino acid sequences of TCRs into a similarity preserved embedding space with contextual understanding of the language of TCRs. Similar TCRs in sequence space have smaller Euclidean distances in vector space while divergent TCRs have larger Euclidean distances. The workflow of the pretraining process is shown below. TCR2vec can also be finetuned for better performance on task-specific data. <br />

<img src="https://github.com/jiangdada1221/TCR2vec_train/blob/main/figures/workflow.jpg" width="800"> <br />

## Dependencies
TCR2vec is writen in Python based on the deeplearning library - Pytorch. Compared to Tensorflow, Pytorch is more user-friendly in __version compatibility__. I would strongly suggest using Pytorch as the deeplearning library so that followers can easily run the code with less pain in making Tensorflow work.  <br />

The required software dependencies are listed below:
 ```
tqdm
scipy
biopython
matplotlib
touch >= 1.1.0 (tested on 1.8.0) 
pandas 
numpy 
sklearn
tape_proteins
 ```

## Installation
 ```
cd TCR2vec
pip install .
 ```

Or you can directly install it as a PyPI package via
```
pip install tcr2vec
```
## Data

 All the source data included in the paper is publicly available, so we suggest readers refer to the original papers for more details. We also uploaded the processed data to google drive which can be accessed via [this link](https://drive.google.com/file/d/1N-88wZYhEs2xaEwFQUn04AaNb8eCgOuv/view?usp=sharing). For the pretraining data, please refer to the [training repository](https://github.com/jiangdada1221/TCR2vec_train).

## Usages of TCR2vec
### Embedding TCRs 
We provide a simple code snip to show how to use TCR2vec for embedding TCRs, which is shown below: <br />

```python
import torch
from tcr2vec.model import TCR2vec
from tape import TAPETokenizer

path_to_TCR2vec = 'path_to_pretrained_TCR2vec'
emb_model = TCR2vec(path_to_TCR2vec)
tokenizer = TAPETokenizer(vocab='iupac') 
#by default, the device for emb_model is cpu
#emb_model = emb_model.to('cuda:0') #to gpu

#example TCR
seq = 'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLRLIYYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV'
token_ids = torch.tensor([tokenizer.encode(seq)])
output = emb_model(token_ids) # shape of 1 x 120

#convert to numpy array
emb = output.detach().cpu().numpy()    

#for a batch input:
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb

dset = TCRLabeledDset([seq],only_tcr=True) #input a list of TCRs
loader = DataLoader(dset,batch_size=32,collate_fn=dset.collate_fn,shuffle=False)
emb = get_emb(emb_model,loader,detach=True) #B x emb_size
```

We also provide a python script *embed.py* in *tcr2vec/* that uses the pretrained model to embed user's input file. The input file should be a csv file, with one column recording the input TCRs (By default, the column name is *full_seq*). 
```
python embed.py --pretrain_path path_to_tcr2vec --dset_path path_to_data.csv --save_path path_to_save_embedding.npy
```
Also, check *python embed.py --h* for more details about input parameters. <br />

### Evaluation of embeddings
The basic script is shown below:
```
python evaluate.py --dset_folder path_to_5fold_dir --pretrain_path path_to_TCRevec --c_method SVM
```
For more experiment settings, pleas enter *python evaluate.py --h* for details.
<br />

### Finetune of TCR2vec
We provide the finetune code for classfication purpose. For writing your custom finetune code, make sure you set the model to training model (*model.train()*)
```
python finetune.py --path_train path_to_train --path_test path_to_test --epoch 20 --batch_size 64 --pretrain_path path_to_TCR2vec --save_path finetune_path.pth 
```
Again, type *python finetune.py --h* for details.
<br />

### Use trained models to make predictions
We provide the code to make prediction scores for TCR-epitope binding using the trained model from either finetuning or using SVM/MLP in *evaluate.py*.
```
python predict.py --dset_path path_to_file --save_prediction_path path_to_save.txt --model_path path_to_finetune.pth
```
Again, type *python predict.py --h* for details. <br />

### Download of pretrained TCR2vec models
[TCR2vec](https://drive.google.com/file/d/1Nj0VHpJFTUDx4X7IPQ0OGXKlGVCrwRZl/view?usp=sharing) <br />
[TCR2vec_large](https://drive.google.com/file/d/1zFE8Kn7gFKzm-RZzcMoy4rn1cIhtIuiU/view?usp=sharing) (embedding size of 768) <br />
[CDR3vec](https://drive.google.com/file/d/1crwG2kLj8O3qQ1zfu8YbEk6cNb5aUD4e/view?usp=sharing) (pretrained on CDR3 sequences) <br />

### Reconstruction of full TCR
The full-length TCR can be recovered by knowing CDR3 + V/J. An example is shown below:
```
python cdr3_to_full_seq.py tcr2vec/data/TCR_gene_segment_data/ tcr2vec/data/sample.csv V J CDR3.beta tcr2vec/data/sample_full.csv 
```
More information can be found on [*cdr3_to_full_seq.py*](https://github.com/jiangdada1221/TCR2vec/blob/main/tcr2vec/cdr3_to_full_seq.py)

### Pretraining of TCR2vec
If you want to re-train TCR2vec on our provided pretraining data or your custom data, please check the [training code](https://github.com/jiangdada1221/TCR2vec_train)

### Notes
* By default, the column names for CDR3, V/J genes, full TCRs are *CDR3.beta*, *V*, *J*, and *full_seq*
* For embdding evaluation, we recommend using the [sklearnx](https://intel.github.io/scikit-learn-intelex/) to accelerate the sklearn models (by specifying *--use_sklearnx True*)
* Example scripts can be found under the [*scripts/*](https://github.com/jiangdada1221/TCR2vec/blob/main/scripts/run.sh) 

## Contact
```
Name: Yuepeng Jiang
Email: yuepjiang3-c@my.cityu.edu.hk/yuj009@eng.ucsd.edu/jiangdada12344321@gmail.com
Note: For instant query, feel free to send me an email since I check email often. Otherwise, you may open an issue section in this repository.
```
Welcome for reporting bugs! 

## License
Free use of TCR2vec is granted under the terms of the GNU General Public License version 3 (GPLv3).

