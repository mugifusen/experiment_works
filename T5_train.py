# プログラムに必要なパッケージ
# pip install torch transformers datasets sentencepiece accelerate
from torch.utils.data import Dataset

class DialogueDataset(Dataset):
    def __init__(self, encodings, labels,toknier):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        tmp = self.encodings[idx]
        tmp_labels = self.labels[idx]
        input = tokenizer(tmp, truncation=True,max_length=300, padding="max_length", return_tensors='pt')
        label = tokenizer(tmp_labels, truncation=True,max_length=300, padding="max_length", return_tensors='pt')
        return {"input_ids":input["input_ids"][0],"attention_mask":input["attention_mask"][0]
              ,"labels":label["input_ids"][0]}


    def __len__(self):
        return len(self.labels)

import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# データセットを格納するディレクトリのパスを定義
data_dir = "./dialogue_data_csv"

# ディレクトリ内のすべてのCSVファイルを読み込み、結合
df_list = []
count = 0
for file_name in os.listdir(data_dir):
    if count == 20:break
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)
    df_list.append(df)
    count += 1

df = pd.concat(df_list, ignore_index=True)
# 欠損値の削除
df.dropna(inplace=True)
# ここらへんに単語減らす機能入れようとしてやめたのでエラー出たら以前のを持ってくる
# 入力と応答で分割
x = df["user"]
y = df["cpu"]

# データを訓練データとテストデータに分割 (x: 入力データ, y: ラベル)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

# プレフィックスを追加
train_x = "対話: " + train_x
test_x = "対話: " + test_x

# DataFrameのインデックスを連続にする
train_x.reset_index(drop=True, inplace=True)
train_y.reset_index(drop=True, inplace=True)
test_x.reset_index(drop=True, inplace=True)
test_y.reset_index(drop=True, inplace=True)

# トークナイザとモデルをロード
tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")

# データセットを作成
train_dataset = DialogueDataset(train_x, train_y,tokenizer)
test_dataset = DialogueDataset(test_x, test_y,tokenizer)

# トレーニングの設定
training_args = TrainingArguments(
    output_dir='./T5_test_result2',
    num_train_epochs=20,
    evaluation_strategy="steps",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    # gradient_accumulation_steps=2,
    # warmup_steps=500,
    weight_decay=0.001, # 1e-5にしてみる、エポック少なくしておく
    logging_steps = 200,
    eval_steps=200
)

# トレーナーを作成し、ファインチューニングを行う
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# モデルを保存
model.save_pretrained('./T5_finetuned2')


