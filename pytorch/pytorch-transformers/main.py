from model import *
from dataloader import *
from utils import *
import argparse


device = "cpu"
#device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
emsize = 200 # 임베딩 차원
nhid = 200 # nn.TransformerEncoder 에서 피드포워드 네트워크(feedforward network) 모델의 차원
nlayers = 2 # nn.TransformerEncoder 내부의 nn.TransformerEncoderLayer 개수
nhead = 2 # 멀티헤드 어텐션(multi-head attention) 모델의 헤드 개수
dropout = 0.2 # 드랍아웃(dropout) 값
bptt = 35

    parser = argparse.ArgumentParser()
    parser.add_argument('-log_file', default='', type=str)
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-data_path', default='../../data/ranked_abs3_fast_b40/WIKI', type=str)
    parser.add_argument('-model_path', default='../../models', type=str)
    parser.add_argument('-vocab_path', default='../../data/spm9998_3.model', type=str)
    parser.add_argument('-train_from', default='', type=str)

    parser.add_argument('-trunc_src_ntoken', default=500, type=int)
    parser.add_argument('-trunc_tgt_ntoken', default=200, type=int)

    parser.add_argument('-emb_size', default=256, type=int)
    parser.add_argument('-enc_layers', default=8, type=int)
    parser.add_argument('-dec_layers', default=1, type=int)
    parser.add_argument('-enc_dropout', default=6, type=float)
    parser.add_argument('-dec_dropout', default=0, type=float)
    parser.add_argument('-enc_hidden_size', default=256, type=int)
    parser.add_argument('-dec_hidden_size', default=256, type=int)
    parser.add_argument('-heads', default=8, type=int)
    parser.add_argument('-ff_size', default=1024, type=int)
    parser.add_argument("-hier", type=str2bool, nargs='?',const=True,default=True)


    parser.add_argument('-batch_size', default=10000, type=int)
    parser.add_argument('-valid_batch_size', default=10000, type=int)
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-lr', default=3, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-seed', default=0, type=int)

    parser.add_argument('-train_steps', default=20, type=int)
    parser.add_argument('-save_checkpoint_steps', default=20, type=int)
    parser.add_argument('-report_every', default=100, type=int)


    # multi-gpu
    parser.add_argument('-accum_count', default=1, type=int)
    parser.add_argument('-world_size', default=1, type=int)
    parser.add_argument('-gpu_ranks', default='0', type=str)

    # don't need to change flags
    parser.add_argument("-share_embeddings", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-share_decoder_embeddings", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument('-max_generator_batches', default=32, type=int)

    # flags for  testing
    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('-test_from', default='../../results', type=str)
    parser.add_argument('-result_path', default='../../results', type=str)
    parser.add_argument('-alpha', default=0, type=float)
    parser.add_argument('-length_penalty', default='wu', type=str)
    parser.add_argument('-beam_size', default=5, type=int)
    parser.add_argument('-n_best', default=1, type=int)
    parser.add_argument('-max_length', default=250, type=int)
    parser.add_argument('-min_length', default=20, type=int)
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-dataset', default='', type=str)
    parser.add_argument('-max_wiki', default=5, type=int)

    # flags for  hier
    # flags.DEFINE_boolean('old_inter_att', False, 'old_inter_att')
    parser.add_argument('-inter_layers', default='0', type=str)

    parser.add_argument('-inter_heads', default=8, type=int)
    parser.add_argument('-trunc_src_nblock', default=24, type=int)

    # flags for  graph


    # flags for  learning
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=8000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-label_smoothing', default=0.1, type=float)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    args.inter_layers = [int(i) for i in args.inter_layers.split(',')]


vocab, train_data, val_data, test_data = load_data(device)
ntokens = len(vocab.stoi) # 단어 사전(어휘집)의 크기

#model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
model = Summarizer(args, word_padding_idx, vocab_size, device, checkpoint)
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
