# まだ機能的に書いてない
# T5のファインチューニングにおける損失とエポック数の関係を示す
# モデルの性能評価の一つ（過学習をしてるかどうか確認）
# これともう一つは性能指標加える

import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルからデータを読み込む
data = pd.read_csv('training_data.csv')

# プロットの準備
plt.figure(figsize=(10, 5))

# 学習ロスのプロット
plt.plot(data['epoch'], data['training_loss'], label='Training Loss')

# 評価ロスのプロット
plt.plot(data['epoch'], data['evaluation_loss'], label='Evaluation Loss')

# グラフのタイトルとラベルの設定
plt.title('Training and Evaluation Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# グラフの表示
plt.show()
