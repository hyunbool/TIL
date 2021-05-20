from model import TransformerModel
from dataloader import *
from utils import *

device = "cpu"
#device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
emsize = 200 # 임베딩 차원
nhid = 200 # nn.TransformerEncoder 에서 피드포워드 네트워크(feedforward network) 모델의 차원
nlayers = 2 # nn.TransformerEncoder 내부의 nn.TransformerEncoderLayer 개수
nhead = 2 # 멀티헤드 어텐션(multi-head attention) 모델의 헤드 개수
dropout = 0.2 # 드랍아웃(dropout) 값
bptt = 35

vocab, train_data, val_data, test_data = load_data(device)
ntokens = len(vocab.stoi) # 단어 사전(어휘집)의 크기

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

"""
모델 실행
"""
# loss 추정시에는 CrossEntropyLoss가 적용
# optimizer로는 SGD를 구현
# 초기 학습률은 5.0
# StepLR: 에포크에 따라서 학습률을 조절하는데 사용
# 학습하는 동안 기울기 폭발을 방지하기 위해 모든 기울기를 함께 스케일링하는 함수인 nn.utils.clip_grad_norm_ 이용

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float("inf")
epochs = 3 # 에포크 수
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model, train_data, optimizer, criterion, scheduler, ntokens, epoch, device)
    val_loss = evaluate(model, val_data, criterion, ntokens, device)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
