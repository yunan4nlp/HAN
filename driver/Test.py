import sys
sys.path.extend(["../../","../","./"])
import time
from driver.Config import *
from Modules.LSTMEncoderModel import *
from driver.Classifier import *
from data.Dataloader import *
import pickle
import random


def write_sents(pred_labels, onebatch, vocab, file):
    label_strs = []
    for label in pred_labels:
        l = label.item()
        pred_label_str = vocab.id2label(l)
        label_strs.append(pred_label_str)

    max_len = len(onebatch)
    assert max_len == len(label_strs)
    for idx in range(max_len):
        file.write(label_strs[idx] + "$#$")
        for word in onebatch[idx][1]:
            file.write(word + " ")
        file.write("\n")

def evaluate(data, classifier, vocab, outputFile):
    start = time.time()
    classifier.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    label_correct, overall = 0, 0
    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, masks, labels = \
            batch_data_variable(onebatch, vocab)
        pred_labels = classifier.predict(words, extwords, masks)
        write_sents(pred_labels, onebatch, vocab, output)
        true_labels = to_tensor(labels)
        label_correct += pred_labels.eq(true_labels).cpu().sum().item()
        overall += len(labels)
    output.close()
    acc = label_correct * 100.0 / overall
    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time = %.2f " % (len(data), during_time))
    return label_correct, overall, acc


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
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    vec = vocab.create_pretrained_embs(config.pretrained_embeddings_file)

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    model = LSTMEncoder(vocab, config, vec)
    model.load_state_dict(torch.load(config.load_model_path))
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    classifier = Classifier(model)
    test_data = read_corpus(config.test_file, vocab)

    label_correct, overall, test_acc = \
        evaluate(test_data, classifier, vocab, config.test_file + '.out')
    print("Test: acc = %d/%d = %.2f" % \
          (label_correct, overall, test_acc))


