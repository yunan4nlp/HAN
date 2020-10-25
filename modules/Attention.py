from modules.Layer import *
from data.Vocab import *
from torch.nn import functional, init

class Attention(nn.Module):
    def __init__(self, att_hidden):
        super(Attention, self).__init__()
        self.MLP = NonLinear(input_size=att_hidden,
                             hidden_size=att_hidden,
                             activation=nn.Tanh())
        self.U = Variable(torch.FloatTensor(att_hidden, 1), requires_grad=True)

        nn.init.xavier_normal_(self.U)


    def forward(self, data, mask):
        b, l, h = data.size()
        hidden = self.MLP(data)
        value = torch.matmul(hidden, self.U).squeeze(-1)

        v = value + (1 - mask) * -1e20
        weight = functional.softmax(v, -1)
        result = data.permute(2, 0, 1) * weight
        result = torch.sum(result, -1)
        result = result.transpose(0, 1)
        return result


class WordAttention(nn.Module):
    def __init__(self, l1_size, att_hidden):
        super(WordAttention, self).__init__()
        self.MLP = NonLinear(input_size=att_hidden,
                             hidden_size=att_hidden,
                             activation=nn.Tanh())
        self.U = Variable(torch.FloatTensor(att_hidden, l1_size), requires_grad=True)
        self.l1_size = l1_size
        nn.init.xavier_normal_(self.U)


    def forward(self, data, mask):
        hidden = self.MLP(data)
        value = torch.matmul(hidden, self.U)

        v = value + (1 - mask.unsqueeze(-1).repeat(1, 1, self.l1_size)) * -1e20
        weight = functional.softmax(v, -2)



        result = data * weight

        result = torch.sum(result, -2)
        result = result.permute(2, 1, 0)
        return result