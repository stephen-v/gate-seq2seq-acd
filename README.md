# requirments

```shell
pytorch
nltk
tqdm
```

# config

(1) Download Pretrained Word embeddings

```shell
https://drive.google.com/drive/folders/189c1i5XESAT6ibJbaQ8z9WJNWA0hUWeE?usp=sharing
```

(2) Config Worde mbeddings Path

conf/sys.config.yaml

```
'glove_pretrain_yelp': '/data/dataset/word2vec',
'glove_pretrain_amazon': '/data/dataset/word2vec',
```

# 1. Data Processing

```shell
python data_processing
```

# 2. Training

2.1 GSM(Our model)

```shell
python run.py --model_name=GSEQ --mode=train --lr=0.0001 --batch_size=4 --attn_hops=50 --mlp_d=100 --data_source=2016 --loss_weight=0.8 --interval=100 --data_category=Laptops
python run.py --model_name=GSEQ --mode=train --lr=0.0001 --batch_size=4 --attn_hops=100 --mlp_d=200 --data_source=2015 --loss_weight=0.8 --interval=100 --data_category=Restaurants
```

2.2 Gen-Seq2Seq(Baseline)

```shell
python run.py --model_name=GSEQ --mode=train --lr=0.0001 --batch_size=4 --data_source=2015 --interval=100 --data_category=Restaurants
```

2.3 Att-Seq2Seq(Baseline)

```shell
python run.py --model_name=ASEQ --mode=train --lr=0.0001 --batch_size=4 --data_source=2015 --interval=100 --data_category=Restaurants
```

# 3. Evaluations

2.1 GSM(Our model)

```shell
python run.py --model_name=GSEQ --mode=eval --lr=0.0001 --batch_size=4 --attn_hops=50 --mlp_d=100 --data_source=2016 --loss_weight=0.8 --interval=100 --data_category=Laptops
python run.py --model_name=GSEQ --mode=eval --lr=0.0001 --batch_size=4 --attn_hops=100 --mlp_d=200 --data_source=2015 --loss_weight=0.8 --interval=100 --data_category=Restaurants
```

2.2 Gen-Seq2Seq(Baseline)

```shell
python run.py --model_name=GSEQ --mode=eval --lr=0.0001 --batch_size=4 --data_source=2015 --interval=100 --data_category=Restaurants
```

2.3 Att-Seq2Seq(Baseline)

```shell
python run.py --model_name=ASEQ --mode=eval --lr=0.0001 --batch_size=4 --data_source=2015 --interval=100 --data_category=Restaurants
```

