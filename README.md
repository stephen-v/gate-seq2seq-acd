# requirments

```shell
pytorch
nltk
tqdm
```

# config 

The pretrain word embeddings from Yelp and Amazon
```shell
https://drive.google.com/drive/folders/189c1i5XESAT6ibJbaQ8z9WJNWA0hUWeE?usp=sharing
```


# 1. Data Processing
```shell
python data_processing
```


# 2. Training
2.1 GSM
```shell
python run.py --model_name=GSEQ --mode=train --lr=0.0001 --batch_size=4 --attn_hops=50 --mlp_d=100 --data_source=2016 --loss_weight=0.8 --interval=100 --data_category=Laptops
python run.py --model_name=GSEQ --mode=train --lr=0.0001 --batch_size=4 --attn_hops=100 --mlp_d=200 --data_source=2015 --loss_weight=0.8 --interval=100 --data_category=Restaurants
```
2.2 Gen-Seq2Seq 

2.3 Att-Seq2Seq

# 3. Evaluations



