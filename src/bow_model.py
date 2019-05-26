import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d



class BoWNet(nn.Module):

    def __init__(self, config, TEXT):
        super(BoWNet, self).__init__()
        self.config = config

        self.use_cuda = config.yes_cuda > 0 and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # self.linear_0_pre = Linear(in_features=transl_input, out_features=transl_output, bias=True)
        # self.linear_0_hyp = Linear(in_features=transl_input, out_features=transl_output, bias=True)

        # EMBEDDING LAYERS
        assert (TEXT.vocab.vectors.size()[1] == config.embedding_dim) , "Embedding dimension assertion."
        self.glove_embedding = nn.Embedding( TEXT.vocab.vectors.size()[0],config.embedding_dim)
        # load GlovVe pertrained embeddings
        self.glove_embedding.weight.data.copy_(TEXT.vocab.vectors)
        # fix embeddings
        self.glove_embedding.weight.requires_grad = False

        # LINEAR LAYERS
        self.conc_hidden_size = config.hidden_size*2

        self.linear_1 = Linear(in_features=self.conc_hidden_size, out_features=self.conc_hidden_size, bias=False)
        self.linear_2 = Linear(in_features=self.conc_hidden_size, out_features=self.conc_hidden_size, bias=False)
        self.linear_3 = Linear(in_features=self.conc_hidden_size, out_features=self.conc_hidden_size, bias=False)

        self.linear_out = Linear(in_features=self.conc_hidden_size, out_features=config.num_classes, bias=False)

        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(self.conc_hidden_size)
        self.dropout = nn.Dropout(p=config.dropout_ln)

    def forward(self, x):

        #premise, premise_len = x.premise[0].shape[1], x.hypothesis[0].shape[1]
        #hypothesis, hypothesis_len =

        cur_batch_size = x.premise[0].shape[1]

        # x.premise ~ (43, 128) ~ (max_sen_len, batch_size)
        # x.hypothesis ~ (20, 128) ~ (max_sen_len, batch_size)

        # 0.GET EMBEDDINGS:

        premise = self.glove_embedding(x.premise[0].to(self.device))
        hypothesis = self.glove_embedding(x.hypothesis[0].to(self.device))

            # 1.Translation layer:

        # premise = torch.tanh(self.linear_0_pre(premise))
        # premise = self.relu(self.linear_0_pre(premise))
        # premise = self.dropout(premise)
        # hypothesis = torch.tanh(self.linear_0_hyp(hypothesis))
        # hypothesis = self.relu(self.linear_0_hyp(hypothesis))
        # hypothesis = self.dropout(hypothesis)

        # 2 SUM EMBEDDINGS:

        # premise ~ (43, 512, 300) ~ (max_seq_len, batch_size, embedding_dim)
        # hypothesis ~ (20, 512, 300) ~ (max_seq_len, batch_size, embedding_dim)

        # . summation over sentence sequence (sequence dim disappears)
        premise = torch.sum(premise, dim=0)
        hypothesis = torch.sum(hypothesis, dim=0)

        # premise and hypothesis = torch.sum(hypothesis,dim=0)  #([512, 300]) ~ (batch_size, embedding_dim)

        # 3 FIT INTO LINEAR LAYERS
        # = torch.cat((premise_sum, hypothesis_sum), 1) #([512, 600]) ~ (batch_size, embedding_dim)

        x = torch.cat((premise, hypothesis), 1)  # ([128, 600]) ~ (batch_size, embedding_dim)

        x = self.relu(self.linear_1(x))  # ([512, 600]) ~ (batch_size, linear_1_out)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.relu(self.linear_2(x))  # ([512, 600]) ~ (batch_size, linear_2_out)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.relu(self.linear_3(x))  # ([512, 600]) ~ (batch_size, linear_3_out)
        x = self.dropout(x)
        x = self.batchnorm(x)

        # return softmax(self.linear_out(x), dim=1) # for cross entropy we dont need softmax - is has it embedded
        return self.relu(self.linear_out(x))
