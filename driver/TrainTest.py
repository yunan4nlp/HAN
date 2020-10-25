import sys
sys.path.extend(["../","./"])
import time
import torch.optim.lr_scheduler
import random
from driver.Config import *
from driver.Classifier import *
import pickle
from modules.DecoderModel import *
from modules.CHANEncoderModel import *
from modules.HANEncoderModel import *
from modules.LSTMEncoderModel import *
from data.Fscore import *


def train(data, dev_data, test_data, classifier, vocab, config):
    enc_optimizer = Optimizer(filter(lambda p: p.requires_grad, classifier.encoder.parameters()), config)
    dec_optimizer = Optimizer(filter(lambda p: p.requires_grad, classifier.decoder.parameters()), config)

    global_step = 0
    best_ACC = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        l1_overall_label_correct, l1_overall_total = 0, 0
        l2_overall_label_correct, l2_overall_total = 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True, True):
            words, extwords, roles, sent_masks, turn_masks = \
                batch_data_variable(onebatch, vocab)

            gold_l1_label, gold_l2_label = batch_label_variable(onebatch, vocab)
            classifier.train()
            classifier.forward(words, extwords, roles, sent_masks, turn_masks)

            l1_loss = classifier.compute_loss(classifier.l1_score, gold_l1_label)
            l2_loss = classifier.compute_loss(classifier.l2_score, gold_l2_label)

            loss = l1_loss + l2_loss
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            l1_label_correct, l1_total = classifier.compute_l1_accuracy(gold_l1_label)
            l2_label_correct, l2_total = classifier.compute_l2_accuracy(gold_l1_label, gold_l2_label)


            l1_overall_label_correct += l1_label_correct
            l1_overall_total += l1_total
            l1_acc = l1_overall_label_correct * 100.0 / l1_overall_total

            l2_overall_label_correct += l2_label_correct
            l2_overall_total += l2_total
            l2_acc = l2_overall_label_correct * 100.0 / l2_overall_total

            during_time = float(time.time() - start_time)
            print("Step:%d, L1_ACC:%.2f, L2_ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                %(global_step, l1_acc, l2_acc, iter, batch_iter, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, classifier.encoder.parameters()), \
                                        max_norm=config.clip)
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, classifier.decoder.parameters()), \
                                         max_norm=config.clip)

                enc_optimizer.step()
                dec_optimizer.step()

                classifier.encoder.zero_grad()
                classifier.decoder.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                dev_l1_overall_correct_num, dev_l1_overall_predict_num, dev_l1_overall_gold_num, \
                dev_l1_p, dev_l1_r, dev_l1_Fscore, \
                dev_l2_overall_correct_num, dev_l2_overall_predict_num, dev_l2_overall_gold_num, \
                dev_l2_p, dev_l2_r, dev_l2_Fscore = \
                    evaluate(dev_data, classifier, vocab, config.dev_file + '.' + str(global_step))
                print("L1 Dev: Precision = %d/%d = %.2f" % \
                      (dev_l1_overall_correct_num, dev_l1_overall_predict_num, dev_l1_p), end=', ')
                print("Recall = %d/%d = %.2f" % \
                      (dev_l1_overall_correct_num, dev_l1_overall_gold_num, dev_l1_r), end=', ')
                print("Fscore = %.2f" %  (dev_l1_Fscore))

                print("L2 Dev: Precision = %d/%d = %.2f" % \
                      (dev_l2_overall_correct_num, dev_l2_overall_predict_num, dev_l2_p), end=', ')
                print("Recall = %d/%d = %.2f" % \
                      (dev_l2_overall_correct_num, dev_l2_overall_gold_num, dev_l2_r), end=', ')
                print("Fscore = %.2f" %  (dev_l2_Fscore))


                test_l1_overall_correct_num, test_l1_overall_predict_num, test_l1_overall_gold_num, \
                test_l1_p, test_l1_r, test_l1_Fscore, \
                test_l2_overall_correct_num, test_l2_overall_predict_num, test_l2_overall_gold_num, \
                test_l2_p, test_l2_r, test_l2_Fscore = \
                    evaluate(test_data, classifier, vocab, config.test_file + '.' + str(global_step))
                print("L1 Test: Precision = %d/%d = %.2f" % \
                      (test_l1_overall_correct_num, test_l1_overall_predict_num, test_l1_p), end=', ')
                print("Recall = %d/%d = %.2f" % \
                      (test_l1_overall_correct_num, test_l1_overall_gold_num, test_l1_r), end=', ')
                print("Fscore = %.2f" %  (test_l1_Fscore))

                print("L2 Test: Precision = %d/%d = %.2f" % \
                      (test_l2_overall_correct_num, test_l2_overall_predict_num, test_l2_p), end=', ')
                print("Recall = %d/%d = %.2f" % \
                      (test_l2_overall_correct_num, test_l2_overall_gold_num, test_l2_r), end=', ')
                print("Fscore = %.2f" %  (test_l2_Fscore))

                if dev_l2_Fscore > best_ACC:
                    print("Exceed best L2 : history = %.2f, current = %.2f" % (best_ACC, dev_l2_Fscore))
                    best_ACC = dev_l2_Fscore
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(classifier.encoder.state_dict(), config.save_encoder_path)
                        torch.save(classifier.decoder.state_dict(), config.save_decoder_path)

def evaluate(data, classifier, vocab, outputFile):
    start = time.time()
    output = open(outputFile, 'w', encoding='utf-8')
    l1_overall_correct_num, l1_overall_predict_num, l1_overall_gold_num = 0, 0, 0
    l2_overall_correct_num, l2_overall_predict_num, l2_overall_gold_num = 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False, False):
        words, extwords, roles, sent_masks, turn_masks = \
            batch_data_variable(onebatch, vocab)

        gold_l1_label, gold_l2_label = batch_label_variable(onebatch, vocab)

        classifier.eval()

        classifier.predict(words, extwords, roles, sent_masks, turn_masks)

        #l1_label_correct, l1_total = classifier.compute_l1_accuracy(gold_l1_label)
        #l2_label_correct, l2_total = classifier.compute_l2_accuracy(gold_l1_label, gold_l2_label)

        predict_labels, gold_labels = get_labels(classifier.l1_score, classifier.l2_score, gold_l1_label, gold_l2_label, onebatch, vocab)

        write_inst(predict_labels, gold_labels, onebatch, output)

        l1_correct_num, l1_predict_num, l1_gold_num,\
        l2_correct_num, l2_predict_num, l2_gold_num = info_stat(predict_labels, gold_labels)

        l1_overall_correct_num += l1_correct_num
        l1_overall_predict_num += l1_predict_num
        l1_overall_gold_num += l1_gold_num

        l2_overall_correct_num += l2_correct_num
        l2_overall_predict_num += l2_predict_num
        l2_overall_gold_num += l2_gold_num

    output.close()

    l1_p, l1_r, l1_Fscore = fscore(l1_overall_correct_num, l1_overall_predict_num, l1_overall_gold_num)
    l2_p, l2_r, l2_Fscore = fscore(l2_overall_correct_num, l2_overall_predict_num, l2_overall_gold_num)
    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time = %.2f " % (len(data), during_time))
    return l1_overall_correct_num, l1_overall_predict_num, l1_overall_gold_num, \
           l1_p, l1_r, l1_Fscore, \
           l2_overall_correct_num, l2_overall_predict_num, l2_overall_gold_num, \
           l2_p, l2_r, l2_Fscore

def write_inst(predict_labels, gold_labels, onebatch, file):
    b = len(onebatch)
    for idx in range(b):
        id = onebatch[idx][0]
        file.write(id + '\n')

        p = predict_labels[idx]

        g = gold_labels[idx]
        if len(p) == 0:
            file.write('nolabel\n')
        else:

            keys = sorted(p.keys())
            for key in keys:
                value = p[key]
                file.write(key + '###' + value + ' ')
            file.write('\n')

        keys = sorted(g.keys())
        for key in keys:
            value = g[key]
            file.write(key + '###' + value + ' ')
        file.write('\n')

        file.write('\n')
    return

class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='classifier.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)


    vocab = creatVocab(config)

    print(vocab._id2role)

    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))


    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    encoder = eval(config.encoder)(vocab, config, vec)
    decoder = Decoder(vocab, config)
    print(encoder)
    print(decoder)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        encoder = encoder.cuda()
        decoder = decoder.cuda()

        #parser = parser.cuda()

    classifier = Classifier(encoder, decoder)

    train_data = read_corpus(config.train_file, config.max_sent_length, config.max_turn_length, vocab)
    dev_data = read_corpus(config.dev_file, config.max_sent_length, config.max_turn_length, vocab)
    test_data = read_corpus(config.test_file, config.max_sent_length, config.max_turn_length, vocab)


    print("train num:", len(train_data))
    print("dev num:", len(dev_data))
    print("test num:", len(test_data))

    train(train_data, dev_data, test_data, classifier, vocab, config)
