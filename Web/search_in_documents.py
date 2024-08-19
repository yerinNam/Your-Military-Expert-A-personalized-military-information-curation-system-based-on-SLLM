import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('jhgan/ko-sroberta-multitask')

def search_in_documents(query, data_path, top_k=2):
    """
    주어진 쿼리에 대해 모든 문서에서 FAISS 인덱스를 사용하여 문장을 검색합니다.

    Parameters:
    - query (str): 검색할 쿼리
    - data_path (str): FAISS 인덱스와 문장이 저장된 디렉터리 경로
    - top_k (int): 검색 결과로 반환할 상위 문장의 수

    Returns:
    - list of tuple: (문장, 거리, 파일 이름)의 리스트
    """
    query_embedding = embedder.encode([query])

    results = []
    for filename in os.listdir(data_path):
        if filename.endswith('_faiss_index.bin'):
            index_file = os.path.join(data_path, filename)
            index = faiss.read_index(index_file)
            
            sentences_file = index_file.replace('_faiss_index.bin', '_sentences.txt')
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences = f.read().splitlines()

            distances, indices = index.search(np.array(query_embedding), top_k)

            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx < len(sentences):
                    result_sentence = sentences[idx]
                    results.append((result_sentence, distances[0][i], filename))
                    
    results = sorted(results, key=lambda x: x[1])
    return results[:top_k]
