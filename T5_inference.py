
# 実行する際にエラー出たら、pip install protobuf==3.20.1
from transformers import T5ForConditionalGeneration, T5TokenizerFast

class TextGenerator:
    def __init__(self):
        model_dir = './T5_finetuned'
        tokenizer_name = "sonoisa/t5-base-japanese"
        model_max_length = 1024

        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name, 
                                                         model_max_length=model_max_length)

    def generate_text(self, input_text):
        max_length = 15
        min_length = 5
        do_sample = True
        temperature = 2.0
        top_k = 80
        top_p = 0.95
        repetition_penalty = 2.5
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # 推論
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature, # 変動を抑制する
            top_k=top_k, # サンプリングプールを狭くする
            top_p=top_p, # nucleus samplingを抑制する
            repetition_penalty=repetition_penalty # 繰り返しのペナルティ
        )

        # 生成されたID列をテキストにデコード
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output

# 使用例
if __name__ == '__main__':
    generator = TextGenerator()
    output_text = generator.generate_text("今日の天気は心地よいね")
    print(output_text)
