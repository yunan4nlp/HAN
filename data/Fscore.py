import sys
sys.path.extend(["../","./"])
from collections import OrderedDict
import argparse
from data.Dataloader import *

def info_stat(batch_pred, batch_gold):
    b = len(batch_gold)

    l1_correct_num = 0
    l1_predict_num = 0
    l1_gold_num = 0

    l2_correct_num = 0
    l2_predict_num = 0
    l2_gold_num = 0
    for idx in range(b):
        inst_predict = batch_pred[idx]
        inst_gold = batch_gold[idx]

        l1_p = to_str_l1(inst_predict)
        l1_g = to_str_l1(inst_gold)
        l1_correct_num += len(l1_p & l1_g)
        l1_predict_num += len(l1_p)
        l1_gold_num += len(l1_g)

        l2_p = to_str_l2(inst_predict)
        l2_g = to_str_l2(inst_gold)
        l2_correct_num += len(l2_p & l2_g)
        l2_predict_num += len(l2_p)
        l2_gold_num += len(l2_g)

    return l1_correct_num, l1_predict_num, l1_gold_num, \
        l2_correct_num, l2_predict_num, l2_gold_num

def to_str_l2(label_dict):
    result = set()
    for key, value in label_dict.items():
        result.add(key + '###' + value)
    return result

def to_str_l1(label_dict):
    result = set()
    for key, value in label_dict.items():
        result.add(key)
    return result

def get_labels(l1_predict, l2_predict, l1_gold, l2_gold, onebatch, vocab):
    l1_pred_labels = l1_predict.data.max(-1)[1].cpu()
    l2_pred_labels = l2_predict.data.max(-1)[1].cpu()

    b = len(onebatch)
    batch_pred = []
    batch_gold = []
    for idx in range(b):
        batch_gold.append(onebatch[idx][-1])
        predict_labels = OrderedDict()
        l1_labels = l1_pred_labels[idx]
        l2_labels = l2_pred_labels[idx]
        l2_gold_labels = l2_gold[idx]
        for idy, l1 in enumerate(l1_labels):
            if l1 == 1:
                assert l2_gold_labels[idy] is not -1
                l1_str = vocab.id2l1(idy)
                l2_str = vocab.id2l2(l2_labels[idy])

                predict_labels[l1_str] = l2_str
        batch_pred.append(predict_labels)
    return batch_pred, batch_gold

def fscore(correct_num, predict_num, gold_num):
    if predict_num == 0:
        p = 0.0
    else:
        p = correct_num * 100.0 / predict_num

    r = correct_num * 100.0 / gold_num

    f = 2 * correct_num  * 100.0 / (predict_num + gold_num)
    return p, r, f

def get_inst(file):
    inst = []
    for line in file.readlines():
        line = line.strip()
        if line == '':
            assert len(inst) == 3
            yield inst
            inst = []
        else:
            inst.append(line)

class LabelsVocab:
    def __init__(self, list):
        reverse = lambda x: dict(zip(x, range(len(x))))
        self.id2label = list
        self.label2id = reverse(list)


class Metric:
    def __init__(self, label):
        self.label = label
        self.correct = 0
        self.predict = 0
        self.gold = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def showFmicro(self):
        print(self.label, end=',\t')
        print('Precision = %d/%d = %.2f' %(self.correct, self.predict, self.precision), end=',\t')
        print('Recall = %d/%d = %.2f' %(self.correct, self.gold, self.recall), end=',\t')
        print('Fscore = %.2f' %self.fscore)

    def showFmacro(self):
        print(self.label, end=',\t')
        print('Precision = %.2f' %(self.precision), end='\t')
        print('Recall = %.2f' %(self.recall), end='\t')
        print('Fscore = %.2f' %self.fscore)

    def PRF(self):
        if self.predict == 0:
            self.precision = 0
        else:
            self.precision = self.correct * 100.0 / self.predict

        if self.gold == 0:
            self.recall = 0
        else:
            self.recall = self.correct * 100.0 / self.gold

        if self.predict + self.gold == 0:
            self.fscore = 0
        else:
            self.fscore = 2 * self.correct * 100.0 / (self.predict + self.gold)

def count4metrics(path):
    L1_list, true_false = build_labels()
    hold = ['涉及', '不涉及']

    L1 = []
    for l1 in L1_list:
        for g in hold:
            L1.append(l1+'###'+g)

    L2 = []
    for l1 in L1_list:
        for g in true_false:
            L2.append(l1 + '###' + g)

    l1_vocab = LabelsVocab(L1)
    l2_vocab = LabelsVocab(L2)

    max_len = len(L1)
    l1_metrics = []
    for idx in range(max_len):
        l1_metrics.append(Metric(L1[idx]))

    max_len = len(L2)
    l2_metrics = []
    for idx in range(max_len):
        l2_metrics.append(Metric(L2[idx]))

    file = open(path, mode='r', encoding='utf8')
    inst_num = 0
    c = 0
    for inst in get_inst(file):
        inst_num += 1
        assert len(inst) == 3
        pred_L1_labels, pred_L2_labels = parse_labels(inst[1], L1_list)
        gold_L1_labels, gold_L2_labels = parse_labels(inst[2], L1_list)

        if '是否虚假诉讼###不涉及' not in pred_L1_labels:
            c += 1

        count(pred_L1_labels, gold_L1_labels, l1_vocab, l1_metrics)
        count(pred_L2_labels, gold_L2_labels, l2_vocab, l2_metrics)
    file.close()
    print("Instance num: ", inst_num)
    for m in l1_metrics:
        m.PRF()

    for m in l2_metrics:
        m.PRF()
    return l1_metrics, l2_metrics

def count(pred_labels, gold_labels, vocab, metrics):
    for pred_l in pred_labels:
        assert pred_l in set(vocab.id2label)
        id = vocab.label2id.get(pred_l)
        metrics[id].predict += 1
        if pred_l in gold_labels:
            metrics[id].correct += 1

    for gold_l in gold_labels:
        assert gold_l in set(vocab.id2label)
        id = vocab.label2id.get(gold_l)
        metrics[id].gold += 1


def each_micro(metrics):
    assert len(metrics)% 2 == 0
    label_micro_metrics = []
    max_len = len(metrics)
    for idx in range(max_len)[::2]:
        m1 = metrics[idx]
        m2 = metrics[idx + 1]
        label1 = m1.label.split('###')[0]
        label2 = m2.label.split('###')[0]
        assert label1 == label2
        m3 = Metric(label1)
        m3.predict = m1.predict + m2.predict
        m3.gold = m1.gold + m2.gold
        m3.correct = m1.correct + m2.correct
        m3.PRF()
        #m3.showFmicro()
        label_micro_metrics.append(m3)
    return label_micro_metrics

def l1_involve_each_micro_metric(metrics):
    assert len(metrics)% 2 == 0
    l1_label_hold_micro_metrics = []
    max_len = len(metrics)
    for idx in range(max_len)[::2]:
        m1 = metrics[idx]
        m2 = metrics[idx + 1]
        label1, involve = m1.label.split('###')
        label2, uninvolve = m2.label.split('###')
        assert label1 == label2
        assert involve == '涉及'
        assert uninvolve == '不涉及'
        l1_label_hold_micro_metrics.append(m1)
    return l1_label_hold_micro_metrics

def each_macro(metrics):
    assert len(metrics)% 2 == 0
    label_macro_metrics = []
    max_len = len(metrics)
    for idx in range(max_len)[::2]:
        m1 = metrics[idx]
        m2 = metrics[idx + 1]
        label1 = m1.label.split('###')[0]
        label2 = m2.label.split('###')[0]
        assert label1 == label2
        m3 = Metric(label1)
        m3.precision = (m1.precision + m2.precision) / 2
        m3.recall = (m1.recall + m2.recall) / 2

        if m3.precision + m3.recall == 0:
            m3.fscore = 0
        else:
            m3.fscore = 2 * m3.precision * m3.recall / (m3.precision + m3.recall)

        #m3.showFmacro()
        label_macro_metrics.append(m3)
    return label_macro_metrics


def parse_labels(line, L1_list):
    line = line.strip()
    info = line.split(' ')
    L2_labels = set(info)

    have_L1 = set()
    L1_labels = set()
    if info[0] == 'nolabel':
        assert len(info) == 1
        L2_labels.clear()
        for l in L1_list:
            L1_labels.add(l + '###' + '不涉及')
    else:
        for labels in info:
            label = labels.split('###')
            key, value = label
            have_L1.add(key)
        for l in L1_list:
            if l in have_L1:
                L1_labels.add(l + '###' + '涉及')
            else:
                L1_labels.add(l + '###' + '不涉及')
    return L1_labels, L2_labels

def macro(metrics, name):
    macro_metric = Metric(name)
    sum_p = 0
    sum_r = 0
    for m in metrics:
        sum_p += m.precision
        sum_r += m.recall
    macro_metric.precision = sum_p / len(metrics)
    macro_metric.recall = sum_r / len(metrics)
    if macro_metric.precision + macro_metric.recall == 0:
        macro_metric.fscore = 0
    else:
        macro_metric.fscore = 2 * macro_metric.precision * macro_metric.recall / (macro_metric.precision + macro_metric.recall)
    return macro_metric

def micro(metrics, name):
    micro_metric = Metric(name)
    for m in metrics:
        micro_metric.gold += m.gold
        micro_metric.predict += m.predict
        micro_metric.correct += m.correct
        #micro_metric.showFmicro()
    micro_metric.PRF()
    return micro_metric


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file', default='from_server/dev.txt_norm.5096')

    args, extra_args = argparser.parse_known_args()

    l1_metrics, l2_metrics = count4metrics(args.file)

    print("==================L1 F==================")
    l1_micro = micro(l1_metrics, "L1 F micro")
    l1_micro.showFmicro()

    l1_macro = macro(l1_metrics, "L1 F macro")
    l1_macro.showFmacro()

    each_l1_micro_metric = each_micro(l1_metrics)
    each_l1_macro_metric = each_macro(l1_metrics)

    each_l1_involve_metrics = l1_involve_each_micro_metric(l1_metrics)
    for m in l1_metrics:
        m.showFmicro()
    print("==================L1 involve==================")

    l1_involve_micro = micro(each_l1_involve_metrics,"L1 involve F micro")
    l1_involve_micro.showFmicro()

    l1_involve_macro = macro(each_l1_involve_metrics,"L1 involve F macro")
    l1_involve_macro.showFmacro()

    print("==================L2 F =================")
    l2_micro = micro(l2_metrics, "L2 F micro")
    l2_micro.showFmicro()

    l2_macro = macro(l2_metrics, "L2 F macro")
    l2_macro.showFmacro()

    each_l2_micro_metric = each_micro(l2_metrics)
    each_l2_macro_metric = each_macro(l2_metrics)

    assert len(each_l1_micro_metric) == len(each_l1_macro_metric) == len(each_l2_micro_metric) == len(each_l2_macro_metric)
    max_len = len(each_l1_micro_metric)
    print("===========L1 L2 PRF copy===========")
    print("%.2f\t%.2f\t%.2f"
          % (l1_micro.precision, l1_micro.recall, l1_micro.fscore), end='\t')
    print("%.2f\t%.2f\t%.2f"
          % (l1_macro.precision, l1_macro.recall, l1_macro.fscore))
    print("%.2f\t%.2f\t%.2f"
          % (l2_micro.precision, l2_micro.recall, l2_micro.fscore), end='\t')
    print("%.2f\t%.2f\t%.2f"
          % (l2_macro.precision, l2_macro.recall, l2_macro.fscore))

    print("===========each class copy===========")
    for idx in range(max_len):
        print("%.2f\t%.2f\t%.2f"
              % (each_l1_micro_metric[idx].precision, each_l1_micro_metric[idx].recall, each_l1_micro_metric[idx].fscore), end='\t')
        print("%.2f\t%.2f\t%.2f"
              % (each_l1_macro_metric[idx].precision, each_l1_macro_metric[idx].recall, each_l1_macro_metric[idx].fscore))

        print("%.2f\t%.2f\t%.2f"
              % (each_l2_micro_metric[idx].precision, each_l2_micro_metric[idx].recall, each_l2_micro_metric[idx].fscore), end='\t')
        print("%.2f\t%.2f\t%.2f"
              % (each_l2_macro_metric[idx].precision, each_l2_macro_metric[idx].recall, each_l2_macro_metric[idx].fscore))
