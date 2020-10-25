from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
import argparse
from collections import OrderedDict

def read_corpus(file_path, max_sent_length, max_turn_length, vocab=None):
    data = []

    names = set()
    inf = open('data/final.name', mode='r', encoding='utf8')
    for line in inf.readlines():
        line = line.strip()
        names.add(line)
    inf.close()

    with open(file_path, 'r', encoding='utf8') as infile:
        for inst in read_instance(infile):

            roles = []
            sents = []
            title = inst[0]
            assert title[-8:] == 'newlabel'

            start = 1

            #print(start)

            for idx, info in enumerate(inst[start:-1]):
                if idx >= max_turn_length: break
                role, sent = info.split('\t')
                role = role_norm(role)
                if role is None: continue
                roles.append(role)
                sent = sent.split(" ")

                words = []
                for w in sent:
                    if w in names:
                        words.append('<NAME>')
                    elif w.isnumeric():
                        words.append('<NUM>')
                    else:
                        words.append(w)
                sents.append(words[:max_sent_length])

            gold_labels = get_labels(inst[-1])
            if gold_labels is not None and len(gold_labels) > 0:
                data.append([title, roles, sents, gold_labels])
    return data

def role_norm(role):
    role = role.replace(' ', '')

    error = set(['是否公开审理', '岸头69号,身份代码', '代理权限', '否公开审理', '以上笔录看过,无误',
                 '是否公开开庭审理', '告知上诉权利', '审判员签名',  '33号,组织机构代码', '公开审理',
                 '庭审次数', '组织机构代码', '上午9', '?借条上有一句话'])

    if role in error:
        return None

    if '被' in role and '代' in role and '原' not in role:
        return '被代'

    if '被告' in role and '原' not in role:
        return '被'

    if '原' in role and '代' in role and '被' not in role:
        return '原代'

    if '原告' in role and '被' not in role:
        return '原'

    if '被' in role and '原' not in role:
        return '被'

    if '原' in role and '被' not in role:
        return '原'

    if '原' in role and '被' in role and '代' in role:
        return '原被代'

    if '原' in role and '被' in role and '代' not in role:
        return '原被'

    if '审' in role and '代' in role:
        return '审代'

    if '书记' in role and '代' in role:
        return '代书记'

    if '书记' in role:
        return '书记'

    if '陪审' in role:
        return '陪'

    if '审' in role:
        return '审'

    if '代' in role:
        return '代'

    if '综上' in role or '总计' in role or '合计' in role or '共计' in role or  '证明' in role:
        return '审'

    return role

def labels_numberize(sentences, vocab):
    for sentence in sentences:
        yield labels2id(sentence, vocab)

def labels2id(sentence, vocab):
    labels = sentence[-1]
    l1_ids, l2_ids = [], []
    for key, value in labels.items():
        l1_id = vocab.l12id(key)
        l2_id = vocab.l22id(value)
        l1_ids.append(l1_id)
        l2_ids.append(l2_id)


    return [l1_ids, l2_ids]

def sentences_numberize(sentences, vocab):
    for sentence in sentences:
        yield sentence2id(sentence, vocab)

def sentence2id(sentence, vocab):
    result = []
    for idx, word in enumerate(sentence[2]):
        wordid = vocab.word2id(word)
        extwordid = vocab.extword2id(word)
        result.append([wordid, extwordid])
    rolesids = vocab.role2id(sentence[1])
    return result, rolesids

def get_labels(label_line):
    labels = label_line.split(' ')

    label_sets = set(labels)
    if '还款事实###是否存在还款行为###否' in label_sets and  \
        '还款事实###是否存在还款行为###支付利息###是' in label_sets:
        label_sets.remove('还款事实###是否存在还款行为###否')

    if '还款事实###是否存在还款行为###否' in label_sets and \
            '还款事实###是否存在还款行为###是' in label_sets:
        label_sets.remove('还款事实###是否存在还款行为###否')

    if '还款事实###是否存在还款行为###支付违约金###是' in label_sets and \
            '还款事实###是否存在还款行为###否' in label_sets:
        label_sets.remove('还款事实###是否存在还款行为###否')

    if '还款事实###是否存在还款行为###支付罚息###是' in label_sets and \
            '还款事实###是否存在还款行为###否' in label_sets:
        label_sets.remove('还款事实###是否存在还款行为###否')

    if '还款事实###是否存在还款行为###支付滞纳金###是' in label_sets and \
            '还款事实###是否存在还款行为###否' in label_sets:
        label_sets.remove('还款事实###是否存在还款行为###否')

    L1, L2 = build_labels()

    L1 = set(L1)
    L2 = set(L2)

    gold_labels = OrderedDict()
    for label in label_sets:
        info = label.split('###')
        assert len(info) >= 3

        if info[1] in L1:
            if info[2] == '否':
                if (gold_labels.get(info[1]) == None or gold_labels[info[1]] == '否'):
                    gold_labels[info[1]] = '否'
                else:
                    return None
            else:
                if (gold_labels.get(info[1]) == None or gold_labels[info[1]] == '是'):
                    gold_labels[info[1]] = '是'
                else:
                    return None
    return gold_labels

def build_labels():


    labels = [
        '是否超过诉讼时效', '是否虚假诉讼', '是否涉及刑事犯罪', '是否有调解协议',
        '是否有和解协议', '是否赌债', '是否借款成立', '是否约定借款期限',
        '是否约定还款期限', '是否借款人对部分借款不知情', '是否共同借款',
        '借款性质', '借贷双方关系', '是否夫妻共同债务', '借款用途',
        '是否借款人转移债务成立', '是否出借人转让债权成立', '是否约定利率',
        '是否约定违约条款', '是否借款人未按约定提供借款', '是否预先扣除借款利息', '是否拒绝履行偿还',
        '是否共同还款', '是否存在还款行为', '尚欠事实', '是否物权担保', '是否保证人担保',
        '是否保证人不承担担保责任', '是否担保人履行代偿责任', '是否超出保证期限',
        '是否约定保证期间', '保证范围', '是否担保人无担保能力'
    ]

    true_false = [
         '是', '否'
    ]

    only_true_false_labels = []
    for elem in labels:
        if elem[:2] == '是否':
            only_true_false_labels.append(elem)

    return only_true_false_labels, true_false


def read_instance(file):
    inst = []
    for line in file:
        line = line.strip()
        if line == "":
            yield inst
            inst = []
        else:
            inst.append(line)

def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, is_sorted=True, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []

    data_size = len(data)
    src_ids = sorted(range(data_size), key=lambda src_id: len(data[src_id][1]), reverse=True)
    sorted_data = [data[src_id] for src_id in src_ids]

    if is_sorted:
        batched_data.extend(list(batch_slice(sorted_data, batch_size)))
    else:
        batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def batch_label_variable(batch, vocab):
    batch_size = len(batch)
    #gold_l1_label = Variable(torch.LongTensor(batch_size, vocab.l1_size).zero_(), requires_grad=False)
    #gold_l2_label = Variable(torch.LongTensor(batch_size, vocab.l1_size).zero_(), requires_grad=False)

    gold_l1_label = np.zeros((batch_size, vocab.l1_size), dtype=int)
    gold_l2_label = np.zeros((batch_size, vocab.l1_size), dtype=int)

    gold_l2_label = gold_l2_label - 1
    b = 0
    for label_index in labels_numberize(batch, vocab):
        l1_label_index = label_index[0]
        l2_label_index = label_index[1]

        for (idx, index) in enumerate(l1_label_index):
            gold_l1_label[b, index] = 1
            gold_l2_label[b, index] = l2_label_index[idx]
        b += 1

    gold_l1_label = torch.from_numpy(gold_l1_label).type(torch.LongTensor)
    gold_l2_label = torch.from_numpy(gold_l2_label).type(torch.LongTensor)
    return gold_l1_label, gold_l2_label


def batch_data_variable(batch, vocab):
    batch_size = len(batch)
    turn_size = -1

    length = -1
    for b in range(0, batch_size):
        if len(batch[b][1]) > turn_size: turn_size = len(batch[b][1])

        for t in range(0, len(batch[b][1])):
            if len(batch[b][2][t]) > length: length = len(batch[b][2][t])

    #words = Variable(torch.LongTensor(batch_size, turn_size, length).zero_(), requires_grad=False)
    #extwords = Variable(torch.LongTensor(batch_size, turn_size, length).zero_(), requires_grad=False)
    #sent_masks = Variable(torch.Tensor(batch_size, turn_size, length).zero_(), requires_grad=False)
    #turn_masks = Variable(torch.Tensor(batch_size, turn_size).zero_(), requires_grad=False)

    words = np.zeros((batch_size, turn_size, length), dtype=int)
    extwords = np.zeros((batch_size, turn_size, length), dtype=int)
    sent_masks = np.zeros((batch_size, turn_size, length), dtype=int)
    turn_masks = np.zeros((batch_size, turn_size), dtype=int)
    roles = np.zeros((batch_size, turn_size), dtype=int)

    b = 0
    for sentence, roles_id in sentences_numberize(batch, vocab):
        idx = 0
        for i, role_id in enumerate(roles_id):
            roles[b, i] = role_id
        for info in sentence:
            idy = 0
            for w in info[0]:
                words[b, idx, idy] = w
                sent_masks[b, idx, idy] = 1
                idy += 1

            idy = 0
            for w in info[1]:
                extwords[b, idx, idy] = w
                idy += 1
            turn_masks[b, idx] = 1
            idx += 1
        b += 1

    words = torch.from_numpy(words).type(torch.LongTensor)
    extwords = torch.from_numpy(extwords).type(torch.LongTensor)
    sent_masks = torch.from_numpy(sent_masks).type(torch.Tensor)
    turn_masks = torch.from_numpy(turn_masks).type(torch.Tensor)

    roles = torch.from_numpy(roles).type(torch.LongTensor)

    return words, extwords, roles, sent_masks, turn_masks

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', default='legal_data/test.txt_norm')
    argparser.add_argument('--dev', default='legal_data/test.txt_norm')
    argparser.add_argument('--test', default='legal_data/test.txt_norm')

    argparser.add_argument('--emb', default='emb/emb.sample')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()

    vocab = creatVocab(args.train, 2)

    vocab.load_pretrained_embs(args.emb)

    train_data = read_corpus(args.train, vocab)
    dev_data = read_corpus(args.dev, vocab)
    test_data = read_corpus(args.test, vocab)


    print("train num:", len(train_data))
    print("dev num:", len(dev_data))
    print("test num:", len(test_data))


    for onebatch in data_iter(train_data, 100, True, True):
        words, extwords, sent_masks, turn_masks = \
            batch_data_variable(onebatch, vocab)

        gold_l1_label, gold_l2_label = batch_label_variable(onebatch, vocab)

