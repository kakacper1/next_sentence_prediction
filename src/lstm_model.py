import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MatchLSTM(nn.Module):
    def __init__(self, config, TEXT):
        super(MatchLSTM, self).__init__()
        self.config = config

        use_cuda = config.yes_cuda > 0 and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        #print('word2vec', word2vec.shape)
        #assert len(word2vec[0]) == config.embedding_dim
        self.word_embed = nn.Embedding(TEXT.vocab.vectors.size()[0], config.embedding_dim, padding_idx=0) # todo: check padding index
        self.word_embed.weight.data.copy_(TEXT.vocab.vectors)
        self.word_embed.weight.requires_grad = False

        self.w_e = nn.Parameter(torch.Tensor(config.hidden_size))
        nn.init.uniform_(self.w_e)

        self.w_s = nn.Linear(in_features=config.hidden_size,
                             out_features=config.hidden_size, bias=False)
        self.w_t = nn.Linear(in_features=config.hidden_size,
                             out_features=config.hidden_size, bias=False)
        self.w_m = nn.Linear(in_features=config.hidden_size,
                             out_features=config.hidden_size, bias=False)
        self.fc = nn.Linear(in_features=config.hidden_size,
                            out_features=config.num_classes)
        self.init_linears()

        self.lstm_prem = nn.LSTM(config.embedding_dim, config.hidden_size)
        self.lstm_hypo = nn.LSTM(config.embedding_dim, config.hidden_size)
        self.lstm_match = nn.LSTMCell(2 * config.hidden_size,
                                      config.hidden_size)

        if config.dropout_fc > 0.:
            self.dropout_fc = nn.Dropout(p=config.dropout_fc)

        self.req_grad_params = self.get_req_grad_params()

    def init_linears(self):
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_t.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.uniform_(self.fc.bias)

    def forward(self, premise, premise_len, hypothesis, hypothesis_len):

        # premise
        premise = premise.to(self.device)
        prem_max_len = premise.size(0)
        premise_len, p_idxes = premise_len.sort(dim=0, descending=True)
        _, p_idx_unsort = torch.sort(p_idxes, dim=0, descending=False)
        premise = premise[:, p_idxes]
        # (max_len, batch_size) -> (max_len, batch_size, embed_dim)
        if self.config.dropout_emb > 0. and self.training:
            premise = F.dropout(self.word_embed(premise),
                                p=self.config.dropout_emb,
                                training=self.training)
        else:
            premise = self.word_embed(premise)
        packed_premise = pack_padded_sequence(premise, premise_len)
        # (max_len, batch_size, hidden_size)
        h_s, (_, _) = self.lstm_prem(packed_premise)
        h_s, _ = pad_packed_sequence(h_s)
        h_s = h_s[:, p_idx_unsort]
        premise_len = premise_len[p_idx_unsort]
        # make it 0
        for batch_idx, pl in enumerate(premise_len):
            h_s[pl:, batch_idx] *= 0.


        # hypothesis
        hypothesis = hypothesis.to(self.device)
        hypothesis_max_len = hypothesis.size(0)
        hypothesis_len, h_idxes = hypothesis_len.sort(dim=0, descending=True)
        _, h_idx_unsort = torch.sort(h_idxes, dim=0, descending=False)
        hypothesis = hypothesis[:, h_idxes]
        # (max_len, batch_size) -> (max_len, batch_size, embed_dim)
        if self.config.dropout_emb > 0. and self.training:
            hypothesis = F.dropout(self.word_embed(hypothesis),
                                   p=self.config.dropout_emb,
                                   training=self.training)
        else:
            hypothesis = self.word_embed(hypothesis)
        packed_hypothesis = pack_padded_sequence(hypothesis, hypothesis_len)
        # (max_len, batch_size, hidden_size)
        h_t, (_, _) = self.lstm_hypo(packed_hypothesis)
        h_t, _ = pad_packed_sequence(h_t)
        h_t = h_t[:, h_idx_unsort]
        hypothesis_len = hypothesis_len[h_idx_unsort]
        for batch_idx, hl in enumerate(hypothesis_len):
            h_t[hl:, batch_idx] *= 0.


        return self.fc()

    def get_req_grad_params(self, debug=False):
        print('#parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
        print('{:,}'.format(total_size))
        return params
