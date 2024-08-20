# Your-Military-Expert-A-personalized-military-information-curation-system-based-on-SLLM
> 국방 인공지능 해커톤 제 2회 우수상: Small LLM을 통한 On-Device성 도메인 특화된 Private ChatBot  Service 구축<br/><br/>
> GPU: NVIDIA DGX A100(80GB) × 2    *슈퍼컴퓨터 데이터센터 지원  <br/>
> 운영체제: Ubuntu 24.04​ <br/>
> 프로그래밍 언어 및 프레임 워크: Python, PyTorch, Transformers​, LoRA <br/>
> 데이터 처리 및 분석: Pandas, NumPy​ 

## 학습 절차
![image](https://github.com/user-attachments/assets/da1c4bca-b94e-4856-9297-7bdb56d78d7c)  <br/> 
Corpus Dataset을 Continual Pretraining.ipnyb에서 아래 부분 코드에 데이터셋 경로를 넣으시면 간단한 전처리 후 학습시킬 수 있습니다. <br/>  <br/>

```python
file_path = 'YOUR_DATASET_PATH'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

sentences = sent_tokenize(text)

def remove_extra_spaces(text):
    return ' '.join(text.split())

data_dict = {'text': [remove_extra_spaces(sentence.strip()) for sentence in sentences if sentence.strip()]}
dataset = Dataset.from_dict(data_dict)
``` 
<br/> 

## 생성 예제 Rag X <br/>

```python
PROMPT = '''사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다. 그리고 반드시 존댓말을 사용해야 합니다.'''

instruction = "LPG 차량을 장기간(1개월 이상) 주차할 때 주의해야 할 사항은 무엇인가요?"

input_text = f"{PROMPT} {instruction}"

messages = [
        {'role': 'system', 'content': PROMPT},
        {'role': 'user', 'content': input_text}
]

input_ids = tokenizer.apply_chat_template(
messages,
add_generation_prompt=True,
return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
        input_ids,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.6,
        top_p=0.7,
        #repetition_penalty=1.5,
        #no_repeat_ngram_size=4,
        eos_token_id=terminators, 
)

response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
print(response_text)
```
<br/>

## 생성 예제 Rag O <br/>
generate_answer_with_rag 함수는 Continual Training.ipnyb에 있습니다. <br/>

```python
query = "Smartstream L3.5 엔진에 추천되는 엔진 오일의 사양은 무엇인가요?"
data_path = "/root/lab_lm/faiss_indexes_정제 전"
answer = generate_answer_with_rag(query, model, tokenizer, data_path)
print("Answer:")
print(answer)
```

## 생성 문장 비교 <br/> 

## 챗봇 Process <br/> 
![image](https://github.com/user-attachments/assets/e3550c75-9b7e-4fef-bc76-5fed69978c76) <br/> 
