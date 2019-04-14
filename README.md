# SemEval 2019 Task 6

## Paper
UM-IU@LING at SemEval-2019 Task 6: Identifying Offensive Tweets Using BERT and SVMs [[arxiv]](https://arxiv.org/abs/1904.03450)

Jian Zhu, Zuoyu Tian, Sandra Kübler
## Bert
run_bert.py
### Usage
`python run_bert.py --task_name one --do_train --do_eval --do_lower_case --data_dir ./training/ --bert_model bert-base-uncased --max_seq_length 80 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 2.0 --output_dir ./output/`
### Dependency
* pytorch 1.0
* [pytorch_pretrained_bert](https://github.com/huggingface/pytorch-pretrained-BERT)

## SVM
run_svm.py


### Feature Selection
The feature selection part is based on the following papers. If you want to use it, please cite one of the papers:
* Kübler, Sandra, Can Liu, and Zeeshan Ali Sayyed. "To use or not to use: Feature selection for sentiment analysis of highly imbalanced data." Natural Language Engineering 24.1 (2018): 3-37
* Liu, Can, Sandra Kübler, and Ning Yu. "Feature selection for highly skewed sentiment analysis tasks." Proceedings of the Second Workshop on Natural Language Processing for Social Media (SocialNLP). 2014.

## Contact


