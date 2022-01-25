# 1. Data Processing
```shell
python data_processing
```


# 2. GSM 
```shell
python run.py --model_name=GSEQ --mode=train --lr=0.0001 --batch_size=4 --attn_hops=50 --mlp_d=500 --data_source=2015 --loss_weight=0.8 --interval=100 --data_category=Restaurants --att=T
python run.py --model_name=GSEQ --mode=train --lr=0.0001 --batch_size=4 --attn_hops=50 --mlp_d=200 --data_source=2016 --loss_weight=0.8 --interval=100 --data_category=Laptops --att=T
```