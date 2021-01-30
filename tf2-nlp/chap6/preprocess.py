import os
import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt

re_filters = "([~.,!?\"':;)(])"

# pad: 패팅 토큰
pad = "<PAD>"

# sos: 시작 토큰
std = "<SOS>"

# end: 종료 토큰
end = "<END>"

# unk: 사전에 없는 단어
unk = "<UNK>"

pad_index = 0
std_index = 1
end_index = 2
unk_index = 3

marker = [pad, std, end, unk]
change_filter = re.compile(re_filters)
max_sequence = 25

# 데이터 불러오는 함수
def load_data(path):
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])
    
    return question, answer

# 단어 사전을 만들기 위해 데이터 전처리한 후 단어 리스트로 만드는 함수
def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(change_filter, "", sentence)
        for word in sentence.split():
            words.append(word)
    
    return [word for word in words if word]


# 한글 텍스트를 토크나이징 하기 위해 형태소로 분리하는 함수
def prepro_like_mortphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)               
                
    return result_data
                           
                           
# 단어 사전을 만드는 함수
def load_vocabulary(path, vocab_path):
    vocabulary_list = []
    
    # 단어 사전이 있으면 그것을 이용
    # 없다면 새로 사전 만들기
    if not os.path.exists(vocab_path):
        if (os.path.exists(path)):
            question, answer = load_data(path)
            data = []
            data.extend(question)
            data.extend(answer)
            
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = marker
            
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
                
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())
            
    word2idx, idx2word = make_vocabulary(vocabulary_list)
    
    return word2idx, idx2word, len(word2idx)
                           

def make_vocabulary(vocabulary_list):
    # 키가 단어이고 값이 인덱스인 딕셔너리 생성
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    
    # 키가 인덱스이고 값이 단어인 딕셔너리 생성
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}
    
    # 두 개의 딕셔너리를 넘겨 준다.
    return word2idx, idx2word

# 인코더에 적용할 입력값 만드는 함수
# value: 전처리 진행할 데이터
# dictionary: 단어 사전
def enc_processing(value, dictionary):
    sequences_input_index = []
    sequences_length = []
    
    for sequence in value:
        # 특수문자 제거
        sequence = re.sub(change_filter, "", sequence)
        sequence_index = []
        
        # 띄어쓰기 기준으로 토크나이징 진행
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[unk]])
            
        # 시퀀스가 max_sequence보다 긴 경우 자르기
        if len(sequence_index) > max_sequence:
            sequence_index = sequence_index[:max_sequence]
            
        sequences_length.append(len(sequence_index))
        
        # max_sequence보다 짧은 경우 <PAD> 넣어줌
        sequence_index += (max_sequence - len(sequence_index)) * [dictionary[pad]]
        
        sequences_input_index.append(sequence_index)
        
    # np.asarray(sequences_input_index: 전처리한 데이터
    # sequences_length: 패딩하기 전 각 문장의 실제 길이 담고 있는 리스트
    return np.asarray(sequences_input_index), sequences_length


# 디코더에 적용할 입력 값 만드는 함수
# 1) 디코더의 입력으로 사용될 입력값 만들기
# 2) 디코더의 결과로 학습을 위해 필요한 라벨인 타깃값 만들기
def dec_output_processing(value, dictionary):
    sequences_output_index = []
    sequences_length = []
    
    for sequence in value:
        # 특수문자 제거
        sequence = re.sub(change_filter, "", sequence)
        sequence_index = []
        sequence_index = [dictionary[std]] + [dictionary[word] for word in sequence.split()]
        
            
        # 시퀀스가 max_sequence보다 긴 경우 자르기
        if len(sequence_index) > max_sequence:
            sequence_index = sequence_index[:max_sequence]
        sequences_length.append(len(sequence_index))
        
        # max_sequence보다 짧은 경우 <PAD> 넣어줌
        sequence_index += (max_sequence - len(sequence_index)) * [dictionary[pad]]
        
        sequences_output_index.append(sequence_index)
        
    # np.asarray(sequences_input_index: 전처리한 데이터
    # sequences_length: 패딩하기 전 각 문장의 실제 길이 담고 있는 리스트
    return np.asarray(sequences_output_index), sequences_length

def dec_target_processing(value, dictionary):
    sequences_target_index = []
    for sequence in value:
        sequence = re.sub(change_filter, "", sequence)
        sequence_index = [dictionary[word] for word in sequence.split()]
        if len(sequence_index) >= max_sequence:
            sequence_index = sequence_index[:max_sequence -1] + [dictionary[end]]
        else:
            sequence_index += [dictionary[end]]
        
        sequence_index += (max_sequence - len(sequence_index)) * [dictionary[pad]]
        sequences_target_index.append(sequence_index)
        
    return np.asarray(sequences_target_index)
       
                           