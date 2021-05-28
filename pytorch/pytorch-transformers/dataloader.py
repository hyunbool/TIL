"""
데이터 로드하고 배치 만들기
"""
# torchtext를 이용해 Wikitext-2 데이터셋 생성
# 단어 오브젝트는 훈련 데이터셋에 의해 만들어지고, 토큰을 텐서로 수치화하는데 사용됨
# 시퀀스 데이터로부터 시작해 batchfy() 함수를 통해 데이터셋을 컬럼들로 배열하고, batch_size의 배치로 나눈 후 남은 모든 토큰을 버림

import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab


def data_process(raw_text_iter, tokenizer, vocab):
    # 각 단어를 단어 사전에서 찾아 텐서로 만듬
    data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                         dtype=torch.long) for item in raw_text_iter]

    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchfy(data, bsz, device):
    # 데이터셋을 bsz 파트들로 나눔
    nbatch = data.size(0) // bsz

    # 깔끔하게 나누어 떨어지지 않는 추가적인 부분(나머지) 잘라내기
    data = data.narrow(0, 0, nbatch * bsz)

    # 데이터에 대해 bsz 배치들로 동등하게 나눔
    data = data.view(bsz, -1).t().contiguous()

    return data.to(device)

def load_data(device):
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in train_iter:
        counter.update(tokenizer(line))
    vocab = Vocab(counter)

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, tokenizer, vocab)
    val_data = data_process(val_iter, tokenizer, vocab)
    test_data = data_process(test_iter, tokenizer, vocab)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchfy(train_data, batch_size, device)
    val_data = batchfy(val_data, eval_batch_size, device)
    test_data = batchfy(test_data, eval_batch_size, device)

    return vocab, train_data, val_data, test_data
