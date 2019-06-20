import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM_for_NSP(nn.Module):
    def __init__(self, config, TEXT):
        super(LSTM_for_NSP, self).__init__()
        self.config = config

        use_cuda = config.yes_cuda > 0 and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        #print('word2vec', word2vec.shape)
        #assert len(word2vec[0]) == config.embedding_dim
        self.word_embed = nn.Embedding(TEXT.vocab.vectors.size()[0], config.embedding_dim, padding_idx=0)
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
        h_s, (pre_hidd, pre_cell) = self.lstm_prem(packed_premise)
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
        h_t, (hyp_hidd, hyp_cell) = self.lstm_hypo(packed_hypothesis)
        h_t, _ = pad_packed_sequence(h_t)
        h_t = h_t[:, h_idx_unsort]
        hypothesis_len = hypothesis_len[h_idx_unsort]
        for batch_idx, hl in enumerate(hypothesis_len):
            h_t[hl:, batch_idx] *= 0.
            # (max_len, batch_size, hidden_size)
        # matchLSTM
        batch_size = premise.size(1)
        h_m_k = torch.zeros((batch_size, self.config.hidden_size),
                            device=self.device)
        c_m_k = torch.zeros((batch_size, self.config.hidden_size),
                            device=self.device)
        h_last = torch.zeros((batch_size, self.config.hidden_size),
                             device=self.device)

        for k in range(hypothesis_max_len):
            h_t_k = h_t[k]

            # Equation (6)
            # (prem_max_len, batch_size)
            e_kj = torch.zeros((prem_max_len, batch_size), device=self.device)
            w_e_expand = \
                self.w_e.expand(batch_size, self.config.hidden_size)\
                    .view(batch_size, 1, self.config.hidden_size)
            for j in range(prem_max_len):
                s_t_m = \
                    torch.tanh(self.w_s(h_s[j]) + self.w_t(h_t_k) +
                               self.w_m(h_m_k))\
                    .view(batch_size, self.config.hidden_size, 1)

                # batch-wise dot product
                # https://discuss.pytorch.org/t/dot-product-batch-wise/9746
                e_kj[j] = torch.bmm(w_e_expand, s_t_m).view(batch_size)

            # Equation (3)
            # (prem_max_len, batch_size)
            alpha_kj = F.softmax(e_kj, dim=0)

            # Equation (2)
            # (batch_size, hidden_size)
            a_k = torch.bmm(alpha_kj.t().view(batch_size, 1, prem_max_len),
                            h_s.permute(1, 0, 2)).view(batch_size,
                                                       self.config.hidden_size)

            # Equation (7)
            # (batch_size, 2 * hidden_size)
            m_k = torch.cat((a_k, h_t_k), 1)

            # Equation (8)
            # (batch_size, hidden_size)
            h_m_k, c_m_k = self.lstm_match(m_k, (h_m_k, c_m_k))

            # handle variable length sequences: hypothesis
            # (batch_size)
            for batch_idx, hl in enumerate(hypothesis_len):
                if k + 1 == hl:
                    h_last[batch_idx] = h_m_k[batch_idx]

        if self.config.dropout_fc > 0:
            h_last = self.dropout_fc(h_last)

        return self.fc(h_last)

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


class LSTM_for_SNLI(nn.Module):
    def __init__(self, config, TEXT):
        super(LSTM_for_SNLI, self).__init__()
        self.config = config

        use_cuda = config.yes_cuda > 0 and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.word_embed = nn.Embedding(TEXT.vocab.vectors.size()[0], config.embedding_dim, padding_idx=0)
        self.word_embed.weight.data.copy_(TEXT.vocab.vectors)
        self.word_embed.weight.requires_grad = False

        self.translation = nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True)

        # initialize all linear
        self.linear_1 = nn.Linear(in_features=config.hidden_size*4,
                             out_features=config.hidden_size*2, bias=True) # maybe change that to true
        self.linear_2 = nn.Linear(in_features=config.hidden_size*2,
                             out_features=config.hidden_size*2, bias=True)
        self.linear_3 = nn.Linear(in_features=config.hidden_size*2,
                             out_features=config.hidden_size*2, bias=True)
        self.linear_out = nn.Linear(in_features=config.hidden_size*2,
                            out_features=config.num_classes)

        self.init_linears()

        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, bidirectional=True)


        self.dropout_0 = nn.Dropout(p=config.dropout)
        self.dropout_1 = nn.Dropout(p=config.dropout)
        self.dropout_2 = nn.Dropout(p=config.dropout)
        self.dropout_3 = nn.Dropout(p=config.dropout)

        self.req_grad_params = self.get_req_grad_params()

        self.relu = nn.ReLU()

        self.batchnorm_pre_0 = nn.BatchNorm1d(num_features=config.hidden_size )
        self.batchnorm_pre_1 = nn.BatchNorm1d(num_features=config.hidden_size )
        self.batchnorm_hyp_0 = nn.BatchNorm1d(num_features=config.hidden_size )
        self.batchnorm_hyp_1 = nn.BatchNorm1d(num_features=config.hidden_size )

        self.batchnorm_1 = nn.BatchNorm1d(num_features=config.hidden_size * 2)
        self.batchnorm_2 = nn.BatchNorm1d(num_features=config.hidden_size * 2)
        self.batchnorm_3 = nn.BatchNorm1d(num_features=config.hidden_size * 2)

        self.softmax = nn.Softmax(dim=1)

    def init_linears(self):

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.uniform_(self.linear_out.bias)

        nn.init.xavier_uniform_(self.translation.weight)
        nn.init.uniform_(self.translation.bias)
        #self.translation.weight.data.fill_(0.01)
        #self.translation.bias.data.fill_(0.01)

    def forward(self, premise, premise_len, hypothesis, hypothesis_len):

        #iter_batch_size = len(premise_len)

        # PREMISE:

        # 1. transfer to CUDA:
        premise = premise.to(self.device)

        # 2. sort examples in descending order
        premise_len, p_idxes = premise_len.sort(dim=0, descending=True)
        _, p_idx_unsort = torch.sort(p_idxes, dim=0, descending=False)
        premise = premise[:, p_idxes]

        # 3. glove
        premise = self.word_embed(premise)

        # 4 Translation
        premise = self.relu(self.translation(premise)) # (max_len, batch_size) -> (max_len, batch_size, embed_dim)

        packed_premise = pack_padded_sequence(premise, premise_len) # (max_len, batch_size, hidden_size)


        h_s, (pre_hidd, _) = self.lstm(packed_premise)

        pre_hidd_unsorted = pre_hidd[:, p_idx_unsort]

        #pre_hidd_unsorted = pre_hidd_unsorted.view(iter_batch_size, self.config.hidden_size*2)
        #pre_hidd_unsorted = (pre_hidd_unsorted)

        #for batch_idx, pl in enumerate(premise_len):
        #    #h_s[pl:, batch_idx] *= 0.
        #    pre_hidd_unsorted[pl:, batch_idx] *= 0.



        # hypothesis
        hypothesis = hypothesis.to(self.device)
        #hypothesis_max_len = hypothesis.size(0)
        hypothesis_len, h_idxes = hypothesis_len.sort(dim=0, descending=True)
        _, h_idx_unsort = torch.sort(h_idxes, dim=0, descending=False)
        hypothesis = hypothesis[:, h_idxes]
        # (max_len, batch_size) -> (max_len, batch_size, embed_dim)
        hypothesis = self.word_embed(hypothesis)

        # Translation
        hypothesis = self.relu(self.translation(hypothesis))

        packed_hypothesis = pack_padded_sequence(hypothesis, hypothesis_len)
        # (max_len, batch_size, hidden_size)
        h_t, (hyp_hidd, _) = self.lstm(packed_hypothesis)


        #hyp_hidd, b =pad_packed_sequence(hyp_hidd) reallY?
        hyp_hidd_unsorted =hyp_hidd[:, h_idx_unsort]

        #hyp_hidd_unsorted = hyp_hidd_unsorted.view(iter_batch_size , self.config.hidden_size*2)
        #hyp_hidd_unsorted = self.batchnorm_hyp(hyp_hidd_unsorted)

        # Second stage , concatenation
        prem_last_forward = self.batchnorm_pre_0(pre_hidd_unsorted[0, :, :])
        prem_last_backward = self.batchnorm_pre_1(pre_hidd_unsorted[1, :, :])

        hypo_last_forward = self.batchnorm_hyp_0(hyp_hidd_unsorted[0, :, :])
        hypo_last_backward = self.batchnorm_hyp_1(hyp_hidd_unsorted[1, :, :])

        # concatenate:
        all_last_seq_out = torch.cat((prem_last_forward, prem_last_backward, hypo_last_forward, hypo_last_backward), 1)
        #all_last_seq_out = torch.cat((pre_hidd_unsorted, hyp_hidd_unsorted), 1)

        x = self.dropout_0(all_last_seq_out)


        x = self.relu(self.linear_1(x))  # ([512, 600]) ~ (batch_size, linear_1_out)
        x = self.dropout_1(x)
        x = self.batchnorm_1(x)

        x = self.relu(self.linear_2(x))  # ([512, 600]) ~ (batch_size, linear_2_out)
        x = self.dropout_2(x)
        x = self.batchnorm_2(x)

        x = self.relu(self.linear_3(x))  # ([512, 600]) ~ (batch_size, linear_3_out)
        x = self.dropout_3(x)
        x = self.batchnorm_3(x)

        # todo keras  activation='softmax' at the end
        # return softmax(self.linear_out(x), dim=1) # for cross entropy we dont need softmax - is has it embedded


        x = self.linear_out(x)
        x = self.softmax(x)
        return x
        # self.tanh(self.linear_out(x))

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

