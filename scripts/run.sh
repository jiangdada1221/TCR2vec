#Recnstruction from CDR3+V+J to full TCR
python ../tcr2vec/cdr3_to_full_seq.py ../tcr2vec/data/TCR_gene_segment_data/ ../tcr2vec/data/sample.csv V J CDR3.beta ../tcr2vec/results/sample_full.csv 

#evaluate
python ../tcr2vec/evaluate.py --dset_folder ../tcr2vec/data/5fold/ --pretrain_path ../tcr2vec/models/TCR2vec_120 --c_method SVM --use_column full_seq --batch_size 64 --record_path ../tcr2vec/results/svm_120_10_ --use_sklearnx True --save_path ../tcr2vec/results/test
python ../tcr2vec/evaluate.py --dset_folder ../tcr2vec/data/5fold/ --pretrain_path ../tcr2vec/models/TCR2vec_120 --c_method MLP --use_column full_seq --batch_size 64 --record_path ../tcr2vec/results/MLP_120_10_ --use_sklearnx True --save_path ../tcr2vec/results/test2
python ../tcr2vec/evaluate.py --dset_folder ../tcr2vec/data/5fold/ --pretrain_path ../tcr2vec/models/TCR2vec_120 --c_method MLP --use_column full_seq --batch_size 64 --record_path ../tcr2vec/results/MLP_120_10_ --use_sklearnx True --save_path ../tcr2vec/results/test2 --epi_encode atchleyae

#finetune
python ../tcr2vec/finetune.py --path_train ../tcr2vec/data/5fold/0_train.csv --path_test ../tcr2vec/data/5fold/0_test.csv --epoch 20 --pretrain_path ../tcr2vec/models/TCR2vec_120 --save_path ../tcr2vec/results/finetune.pth --save_prediction_path ../tcr2vec/results/finetune_0.txt --batch_size 64

#embed
python ../tcr2vec/embed.py --pretrain_path ../tcr2vec/models/TCR2vec_120 --dset_path ../tcr2vec/data/5fold/0_train.csv --save_path ../tcr2vec/results/embedding.npy

#make predictions from trained model
python ../tcr2vec/predict.py --dset_path ../tcr2vec/data/5fold/0_test.csv --save_prediction_path ../tcr2vec/results/pres.txt --model_path ../tcr2vec/results/test_0.pickle --pretrain_path ../tcr2vec/models/TCR2vec_120
python ../tcr2vec/predict.py --dset_path ../tcr2vec/data/5fold/0_test.csv --save_prediction_path ../tcr2vec/results/pres_finetune.txt --model_path ../tcr2vec/results/finetune.pth