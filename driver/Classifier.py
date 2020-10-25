import torch.nn.functional as F
import torch.optim.lr_scheduler
from modules.Layer import *
import numpy as np


def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def to_tensor(list):
    return torch.from_numpy(np.asarray(list, dtype=np.long)).type(torch.LongTensor)

class Classifier(object):
    def __init__(self, model, decoder):
        self.encoder = model
        self.decoder = decoder

        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        if self.use_cuda:
            self.encoder.sent_att.U = self.encoder.sent_att.U.cuda()
            self.encoder.turn_att.U = self.encoder.turn_att.U.cuda()
            self.decoder.l1_represents = self.decoder.l1_represents.cuda()


    def forward(self, words, extwords, roles, sent_masks, turn_masks):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device),
            roles = roles.cuda(self.device)
            sent_masks = sent_masks.cuda(self.device)
            turn_masks = turn_masks.cuda(self.device)

        d_hidden = self.encoder.forward(words, extwords, roles, sent_masks, turn_masks)
        l1_score, l2_score = self.decoder.forward(d_hidden)
        self.l1_score = l1_score
        self.l2_score = l2_score
        # cache

    def compute_loss(self, pred_labels, gold_labels):
        if self.use_cuda:
            gold_labels = gold_labels.cuda()
        pred_labels = pred_labels.view(-1, 2)
        gold_labels = gold_labels.view(-1)
        loss = F.cross_entropy(
            pred_labels, gold_labels, ignore_index = -1)
        return loss

    def compute_l1_accuracy(self, l1_true_labels):
        pred_labels = self.l1_score
        b, l1_num, _ = pred_labels.size()
        pred_labels = pred_labels.data.max(-1)[1].cpu()
        label_correct = pred_labels.eq(l1_true_labels).cpu().sum()
        return label_correct.item(), b * l1_num

    def compute_l2_accuracy(self, l1_true_labels, l2_true_labels):
        pred_labels =self.l2_score
        pred_labels = pred_labels.data.max(-1)[1].cpu()
        l2_true_labels = l2_true_labels.cpu()

        predict = pred_labels.masked_select(l1_true_labels.type(torch.ByteTensor))
        gold = l2_true_labels.masked_select(l1_true_labels.type(torch.ByteTensor))

        label_correct = predict.eq(gold).cpu().sum()
        return label_correct.item(), predict.size()[0]

    def predict(self, words, extwords, roles, sent_masks, turn_masks):
        if words is not None:
            self.forward(words, extwords, roles, sent_masks, turn_masks)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
