import cupy as cp
import pandas as pd
import pandas as pd


train_df = pd.read_csv("news_article/train.csv")
test_df = pd.read_csv("news_article/test.csv")

train_df = train_df.dropna(axis=1)
test_df = test_df.dropna(axis=1)
