## Requirements

- torch==1.7.0
- transformers==3.2.0

## Preparation

Download and unzip GloVe vectors(`glove.840B.300d.zip`) from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and put it into  `/glove` directory.

## build vocab for different datasets

python ./prepare_vocab.py --data_dir dataset/Restaurants_corenlp --vocab_dir dataset/Restaurants_corenlp
python ./prepare_vocab.py --data_dir dataset/Laptops_corenlp --vocab_dir dataset/Laptops_corenlp
python ./prepare_vocab.py --data_dir dataset/Tweets_corenlp --vocab_dir dataset/Tweets_corenlp


## Training

python ./train.py --model_name cdlm --dataset restaurant  --num_epoch 50 --vocab_dir ./dataset/Restaurants_corenlp --cuda 0 --num_layer 2

python ./train.py --model_name cdlm --dataset laptop  --num_epoch 50 --vocab_dir ./dataset/Laptops_corenlp --cuda 0 --num_layer 2

python ./train.py --model_name cdlm --dataset twitter  --num_epoch 50 --vocab_dir ./dataset/Tweets_corenlp --cuda 0 --num_layer 2
