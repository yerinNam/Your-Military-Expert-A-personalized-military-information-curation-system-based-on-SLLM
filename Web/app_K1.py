from flask import Flask, request, jsonify, render_template, redirect, url_for
import requests
import os
import numpy as np
from search_in_documents import search_in_documents  # 검색 함수 불러오기
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

app = Flask(__name__)

model, tokenizer, data_path = None, None, None

chat_history = []

def load_staria_model():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    device = torch.device('cuda:0')

    model_path = '/root/lab_lm/model/continue_train_ko-bllossom-정비_model_3epochs'
    tokenizer_path = '/root/lab_lm/model/continue_train_ko-bllossom-정비_model_3epochs'
    data_path = '/root/lab_lm/function/faiss_indexes2'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
    base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True).to(device)
    print('Staria LoRA model is loaded on:', model.device)

    return model, tokenizer, data_path

def load_model(model_name):
    global model, tokenizer, data_path  # 전역 변수로 설정

    if model_name == "universe":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        device = torch.device('cuda:0')

        model_path = '/root/lab_lm/model/continue_train_ko-bllossom-universe_model_3epochs'
        tokenizer_path = '/root/lab_lm/model/continue_train_ko-bllossom-universe_model_3epochs'
        data_path = '/root/lab_lm/function/faiss_indexes_universe'

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
        base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True).to(device)
        print('Universe LoRA model is loaded on:', model.device)
    else:
        print("Loading default Staria model...")
        model, tokenizer, data_path = load_staria_model()

    return model, tokenizer, data_path

model, tokenizer, data_path = load_staria_model()

@app.route('/load_model/<model_name>')
def load_selected_model(model_name):
    global model, tokenizer, data_path
    model, tokenizer, data_path = load_model(model_name)
    chat_history = []
    return redirect(url_for('new_page')), chat_history

def get_response_from_model(query, history):
    try:
        device = torch.device('cuda:0')
        print(data_path)
        retrieved_sentences = [result[0] for result in search_in_documents(query, data_path, top_k=1)]
        reference = "\n".join(retrieved_sentences)
        print(f"Retrieved reference: {reference}")

        PROMPT = '''당신은 정비 메뉴얼에 관한 챗봇입니다. 사용자에게 친절하게 답변해주시고 만약 검색된 외부지식이 포함될 경우 외부지식을 활용해서 답변해주세요. 반드시 존댓말을 하세요.'''
        #history = "\n".join([msg['content'] for msg in history if isinstance(msg, dict) and 'content' in msg])
        #print(":::history", history.splitlines()[0], ':::')
        # 히스토리를 포함한 프롬프트 작성
        #formatted_query = f"### human\n\n{history.splitlines()[0]}\n\n### 외부지식\n\n{reference}\n\n### 질의\n\n{query}"
        formatted_query = f"### 외부지식\n\n{reference}\n\n### 질의\n\n{query}"
        formatted_query = [
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': formatted_query}
            ]

        terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        inputs = tokenizer.apply_chat_template(formatted_query, add_generation_prompt=True, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=800, do_sample=True, temperature=0.6, top_p=0.7, eos_token_id=terminators)

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": response_text}]
        
    except Exception as e:
        error_message = f"Exception in get_response_from_model: {e}"
        print(error_message)
        return {"error": error_message}

@app.route('/chat', methods=['POST'])
def chat():
    if len(chat_history) > 2:
        chat_history.pop(0)  # 가장 오래된 대화 기록을 제거하여 두 개만 유지

    data = request.get_json()
    user_input = data['prompt']
        
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append(f"{user_input}")    
                        
    response = get_response_from_model(user_input, chat_history)
    print(f"Response from model: {response}")

    if 'error' in response:
        raise Exception(response['error'])
        
    formatted_response = format_response(response)
        
    # 대화 기록에 챗봇 응답 추가
        
    return jsonify({'response': formatted_response, 'history': chat_history})

def format_response(response):  
    response_text = response[0]['generated_text']
    if 'assistant' in response_text:
        main_response = response_text.split('assistant')[1].strip()
        if '### human' in main_response:
            main_response = response_text.split('### human')[1].strip()
    else:
        main_response = response_text.strip()
    
    if 'bot:' in main_response and 'user:' in main_response:
        main_response = main_response.split('bot:')[1].split('user:')[0].strip()
    
    if '(외부지식 참고)' in main_response:
        main_response = main_response.split('(외부지식)')[0].strip()

    if '---' in main_response:
        main_response = main_response.split('---')[0].strip()    

    return main_response

@app.route('/')
def index():
    return render_template('정비_index.html')

@app.route("/new_page")
def new_page():
    return render_template('new_page.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)





