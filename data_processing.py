import random

from src.data_processing.DataGenerator import DataGenerator
from src.data_processing.EmbeddingGenerator import EmbeddingGenerator
from src.utils.const import SEED
from monai.utils import set_determinism

set_determinism(seed=SEED)

# Dataset contains
# 2015 Restaurants
# 2016 Laptops

data_source = '2016'
category = 'Laptops'

data_generator = DataGenerator(data_source, category)
word2id, cate2id = data_generator.generate_data()
embedding_generator = EmbeddingGenerator(data_source, 300, category)
embedding_generator.generate_embedding(word2id)
