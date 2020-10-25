from modules.Layer import *
from data.Vocab import *


class Decoder(nn.Module):
    def __init__(self, vocab, config):
        super(Decoder, self).__init__()
        self.config = config
        self.l1_size = vocab.l1_size
        self.l1_represents = Variable(torch.FloatTensor(vocab.l1_size, config.lstm_hiddens * 2), requires_grad=True)
        nn.init.xavier_normal_(self.l1_represents)

        #self.l2_represents = Variable(torch.FloatTensor(vocab.l1_size, config.lstm_hiddens * 2), requires_grad=True)
        #nn.init.xavier_normal_(self.l2_represents)

        self.l1_mlp = NonLinear(input_size=config.lstm_hiddens * 2,
                                hidden_size=config.lstm_hiddens,
                                activation=nn.SELU())

        self.l2_mlp = NonLinear(input_size=config.lstm_hiddens * 2,
                                hidden_size=config.lstm_hiddens,
                                activation=nn.SELU())

        self.l1_linear = nn.Linear(in_features=config.lstm_hiddens,
                                   out_features=2,
                                   bias=False)

        self.l2_linear = nn.Linear(in_features=config.lstm_hiddens,
                                   out_features=2,
                                   bias=False)


    def forward(self, d_hidden):
        d_hidden = d_hidden.unsqueeze(1) # b, 1, h
        l1_out = self.l1_score(d_hidden)
        l2_out = self.l2_score(d_hidden)
        return l1_out, l2_out

    def l1_score(self, d_hidden):
        d_hidden = d_hidden.repeat(1, self.l1_size, 1) # b, l1, h
        hidden = torch.mul(d_hidden,  self.l1_represents) # (b, l1, h) * (l1, h)
        l1_mlp_hidden = self.l1_mlp(hidden)
        l1_out = self.l1_linear(l1_mlp_hidden)
        return l1_out

    def l2_score(self, d_hidden):
        d_hidden = d_hidden.repeat(1, self.l1_size, 1) # b, l1, h
        hidden = torch.mul(d_hidden,  self.l1_represents) # (b, l2, h) * (l1, h)
        l2_mlp_hidden = self.l2_mlp(hidden)
        l2_out = self.l2_linear(l2_mlp_hidden)
        return l2_out



