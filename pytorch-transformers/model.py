# nn.TransformerEncoder 모델을 언어 모델링 과제에 대해 학습시켜보자
# 언어 모델링: 주어진 단어(또는 단어 시퀀스)가 다음에 이어지는 단어 시퀀스를 따를 가능성(likelihood)에 대한 확률을 할당
# 토큰들의 시퀀스가 임베딩 레이어로 전달 => 이어서 포지셔널 인코딩 레이어가 각 안어의 순서를 설명
# nn.TransformerEncoder는 여러개의 nn.TransformerEncoderLayer 레이어로 구성
# nn.TransformerEncoder 내부의 셀프 어텐션 레이어들은 시퀀스 안에서 이전 포지션에 집중하도록 만들어지기 때문에 어텐션 마스크를 필요로 함
# 그런 다음 nn.TransformerEncoder의 출력은 로그-소프트맥스로 이어지는 최종 linear 레이어로 전달
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


# PositionalEncoding: 시퀀스 안에서 토큰의 상대적/절대적 포지션에 대한 정보를 주입
# 인베딩과 똑같은 차원을 가지며, sine & cosine 함수를 사용해 표현
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
