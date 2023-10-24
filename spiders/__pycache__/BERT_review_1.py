import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
#from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')


df = pd.read_csv("tagged_data.csv")

# deleting additional information
df = df.drop([df.columns[0], df.columns[2], df.columns[3], df.columns[4]], axis=1)
print(df.head())

# split data
train_df, val_df = train_test_split(df, test_size=0.05)
print(train_df.shape, val_df.shape)

# the distribution of the tags
LABEL_COLUMNS = df.columns.tolist()[2:]
df[LABEL_COLUMNS].sum().sort_values().plot(kind="barh")
plt.show()

# the distribution of the correct and incorrect tags
train_correct = train_df[train_df.iloc[:, 30] == 1]
train_incorrect = train_df[train_df.iloc[:, 31] == 1]
pd.DataFrame(dict(
  correct=[len(train_correct)],
  incorrect=[len(train_incorrect)]
)).plot(kind='barh')
plt.show()

# output of data quantity
count_ones = len(train_df[train_df.iloc[:, 30] == 1])
count_ones_2 = len(train_df[train_df.iloc[:, 31] == 1])
column_name = train_df.columns[30]
column_name_2 = train_df.columns[31]
print(column_name, count_ones, column_name_2, count_ones_2)

# sample only 200 correct reviews to combat the imbalance
train_df = pd.concat([
  train_incorrect,
  train_correct.sample(200)
])
train_df.shape, val_df.shape

# Tokenization
BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# a simple example
sample_row = df.iloc[7]
sample_comment = sample_row.reviews
sample_labels = sample_row[LABEL_COLUMNS]
print(sample_comment)
print()
print(sample_labels.to_dict())

encoding = tokenizer.encode_plus(
  sample_comment,
  add_special_tokens=True,
  max_length=512,
  return_token_type_ids=False,
  padding="max_length",
  return_attention_mask=True,
  return_tensors='pt',
)
encoding.keys()
print(encoding.keys())  # delete

encoding["input_ids"].shape, encoding["attention_mask"].shape
print(encoding["input_ids"].shape)  # delete
print(encoding["attention_mask"].shape)  # delete

encoding["input_ids"].squeeze()[:20]
encoding["attention_mask"].squeeze()[:20]
print(encoding["input_ids"].squeeze()[:20])  # delete
print(encoding["attention_mask"].squeeze()[:20])  # delete

print(tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())[:20])
# the end of the example

# the number of tokens per comment
token_counts = []
for _, row in train_df.iterrows():
  token_count = len(tokenizer.encode(
    row["reviews"],
    max_length=512,
    truncation=True
  ))
  token_counts.append(token_count)
sns.histplot(token_counts)
plt.xlim([0, 512])
plt.show()

MAX_TOKEN_COUNT = 512

# Dataset
class CorrectReviewsDataset(Dataset):
  def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: BertTokenizer,
    max_token_len: int = 128
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
  def __len__(self):
    return len(self.data)
  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]
    reviews = data_row.reviews
    labels = data_row[LABEL_COLUMNS]
    encoding = self.tokenizer.encode_plus(
      reviews,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return dict(
      reviews=reviews,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      #labels = torch.FloatTensor(labels.iloc[0]) #debug but tensor([])
      labels = torch.FloatTensor(labels)
    )

# a sample item from the dataset
train_dataset = CorrectReviewsDataset(
  train_df,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)
sample_item = train_dataset[0]

print(sample_item.keys())
print(sample_item["reviews"])
print(sample_item["labels"])
print(sample_item["input_ids"].shape)

# BERT
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
sample_batch = next(iter(DataLoader(train_dataset, batch_size=4, num_workers=1)))
sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape
output = bert_model(sample_batch["input_ids"], sample_batch["attention_mask"])
output.last_hidden_state.shape, output.pooler_output.shape
bert_model.config.hidden_size