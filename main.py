from bert import EmotionAnalyzer
from T5_inference import TextGenerator

# クラスのインスタンス生成
TextGen = TextGenerator()
EAnalyze = EmotionAnalyzer()

print("Start Nagomi Chat!\n")
print("やあやあNagomiだよ～")

# 過去の対話を保存するためのリストを初期化
past_dialogue = []

while True:
    # T5にユーザの入力を渡す
    user_input = input()

    if user_input == "じゃあね":
        print("またねー")
        print("End Nagomi Chat")
        break

    # ユーザの入力を過去の対話に追加
    past_dialogue.append(user_input)
    # 過去の対話を一つの文字列に結合
    dialogue_text = " ".join(past_dialogue)

    # ユーザの入力を渡す
    text = TextGen.generate_text(dialogue_text)

    # モデルの応答を過去の対話に追加
    past_dialogue.append("Nagomi: " + text)

    # 感情分析して絵文字を取得
    emotion = EAnalyze.analyze_emotion(text)
    emoji = EAnalyze.select_emoji(emotion)
    print(text + emoji)
