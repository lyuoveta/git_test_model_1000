import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from torchmetrics import F1Score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn import metrics
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import multiprocessing
multiprocessing.freeze_support()
import warnings



df = pd.read_csv("tagged_data.csv")

# deleting additional information
df = df.drop([df.columns[0], df.columns[2], df.columns[3], df.columns[4],df.columns[6], df.columns[7], df.columns[8],
              df.columns[9], df.columns[10], df.columns[11],df.columns[12], df.columns[13], df.columns[14],df.columns[15],
              df.columns[16], df.columns[17], df.columns[18], df.columns[19], df.columns[20], df.columns[21], df.columns[22],
             df.columns[23],df.columns[24], df.columns[25], df.columns[26],df.columns[27], df.columns[28], df.columns[29],df.columns[30],
              df.columns[31], df.columns[32], df.columns[33], df.columns[36]], axis=1)
print(df.head())

# split data
train_df, val_df = train_test_split(df, test_size=0.05)
print(train_df.shape, val_df.shape)

# the distribution of the tags
LABEL_COLUMNS = df.columns.tolist()[2:]
df[LABEL_COLUMNS].sum().sort_values().plot(kind="barh")
plt.show()

# the distribution of the correct and incorrect tags
train_correct = train_df[train_df.iloc[:, 2] == 1]
train_incorrect = train_df[train_df.iloc[:, 3] == 1]
pd.DataFrame(dict(
  correct=[len(train_correct)],
  incorrect=[len(train_incorrect)]
)).plot(kind='barh')
plt.show()

# output of data quantity
count_ones = len(train_df[train_df.iloc[:, 2] == 1])
count_ones_2 = len(train_df[train_df.iloc[:, 3] == 1])
column_name = train_df.columns[2]
column_name_2 = train_df.columns[3]
print(column_name, count_ones, column_name_2, count_ones_2)

# sample only 200 correct reviews to combat the imbalance
train_df = pd.concat([
  train_incorrect,
  train_correct.sample(200)
])
print(train_df.shape, val_df.shape)

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
print(encoding.keys())
print(encoding["input_ids"].shape)
print(encoding["attention_mask"].shape)
print(encoding["input_ids"].squeeze()[:20])
print(encoding["attention_mask"].squeeze()[:20])
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
warnings.filterwarnings("ignore", category=FutureWarning)


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
        #labels = torch.FloatTensor(labels)
        labels=torch.FloatTensor(labels.iloc[:])
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
BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
sample_batch = next(iter(DataLoader(train_dataset, batch_size=8, num_workers=0)))  # must be 2 num_workers
print(sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape)
output = bert_model(sample_batch["input_ids"], sample_batch["attention_mask"])
print(output.last_hidden_state.shape, output.pooler_output.shape)
print(bert_model.config.hidden_size)

# custom dataset into a LightningDataModule


class CorrectReviewsDataModule(pl.LightningDataModule):
  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
  def setup(self, stage=None):
    self.train_dataset = CorrectReviewsDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )
    self.test_dataset = CorrectReviewsDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=0
    )
  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )
  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )


# an instance of data module
N_EPOCHS = 10
BATCH_SIZE = 12
data_module = CorrectReviewsDataModule(
  train_df,
  val_df,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_TOKEN_COUNT
)

# MODEL
class CorrectReviewsTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        for i, name in enumerate(LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
            )
        )


# Optimizer scheduler
dummy_model = nn.Linear(10, 1)
for param in dummy_model.parameters():
    param.requires_grad = True
optimizer = AdamW(params=dummy_model.parameters(), lr=0.001)
warmup_steps = 20
total_training_steps = 100
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=warmup_steps,
  num_training_steps=total_training_steps
)
learning_rate_history = []
for step in range(total_training_steps):
    optimizer.step()
    scheduler.step()
    learning_rate_history.append(optimizer.param_groups[0]['lr'])
plt.plot(learning_rate_history, label="learning rate")
plt.axvline(x=warmup_steps, color="red", linestyle=(0, (5, 10)), label="warmup end")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Learning rate")
plt.tight_layout()
plt.show()

steps_per_epoch = len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5
print(warmup_steps, total_training_steps)

# an instance of the model
model: CorrectReviewsTagger = CorrectReviewsTagger(
  n_classes=len(LABEL_COLUMNS),
  n_warmup_steps=warmup_steps,
  n_training_steps=total_training_steps
)

# Evaluation
criterion = nn.BCELoss()
prediction = torch.FloatTensor(
  [10.95873564, 1.07321467, 1.58524066, 0.03839076, 15.72987556, 1.09513213]
)
labels = torch.FloatTensor(
  [1., 0., 0., 0., 1., 0.]
)
print(torch.sigmoid(prediction))
print(criterion(torch.sigmoid(prediction), labels))

_, predictions = model(sample_batch["input_ids"], sample_batch["attention_mask"])
print(predictions)
print(criterion(predictions, sample_batch["labels"]))

# ROC Curve
fpr = [0.        , 0.        , 0.        , 0.02857143, 0.02857143,
       0.11428571, 0.11428571, 0.2       , 0.4       , 1.        ]
tpr = [0.        , 0.01265823, 0.67202532, 0.76202532, 0.91468354,
       0.97468354, 0.98734177, 0.98734177, 1.        , 1.        ]
_, ax = plt.subplots()
ax.plot(fpr, tpr, label="ROC")
ax.plot([0.05, 0.95], [0.05, 0.95], transform=ax.transAxes, label="Random classifier", color="red")
ax.legend(loc=4)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Example ROC curve")
plt.show()

# Training

checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="correct-reviews")

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

trainer = pl.Trainer(
  logger=logger,
  callbacks=[early_stopping_callback, checkpoint_callback],
  max_epochs=N_EPOCHS,
  accelerator=None,  # changed, needed CUDA!
  progress_bar_refresh_rate=30
)

trainer.fit(model, data_module)

trainer.test()


# Predictions
trained_model = CorrectReviewsTagger.load_from_checkpoint(
  trainer.checkpoint_callback.best_model_path,
  n_classes=len(LABEL_COLUMNS)
)
trained_model.eval()
trained_model.freeze()
test_comment = "I have been taking rhodiola for over a year.  It works great for reducing general anxiety.  If I go longer than two days without it, I get an inner jitteriness and feelings of doom.  I have noticed no side effects."
encoding = tokenizer.encode_plus(
  test_comment,
  add_special_tokens=True,
  max_length=512,
  return_token_type_ids=False,
  padding="max_length",
  return_attention_mask=True,
  return_tensors='pt',
)
_, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
test_prediction = test_prediction.flatten().numpy()
for label, prediction in zip(LABEL_COLUMNS, test_prediction):
  print(f"{label}: {prediction}")