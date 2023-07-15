# インストールするパッケージ
# pip install torch torchvision transformers fugashi ipadic xformers
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer

# 感情分析と絵文字を選択する
class EmotionAnalyzer:
    def __init__(self):
        # 1. 学習済みモデルの準備
        self.model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
        # 2. 日本語の単語分解
        self.tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        # 3. 感情分析モデルの生成
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def analyze_emotion(self, text):
        emotion_data = self.nlp(text)
        return emotion_data

    def select_emoji(self, emotion_data):
        if not emotion_data:    # 感情分析の結果がNEUTRALの時
            return None
        score = emotion_data[0]["score"]
        label = emotion_data[0]["label"]

        if label == "POSITIVE":
            if score <= 0.5:
                return "😊"
            else:
                return "🥰"
        
        elif label == "NEGATIVE":
            if score <= 0.5:
                return "🙄"
            else:
                return "😭"
        else:
            # ニュートラルの時
            return "🤭"

if __name__=='__main__':
    text = "美味しいごはんが好きなんだー"
    analyzer = EmotionAnalyzer()

    # テキストの感情分析を実行
    emotion_data = analyzer.analyze_emotion(text)
    emoji = analyzer.select_emoji(emotion_data)
    print(emoji)