# coding=utf-8
from __future__ import print_function
import optparse
import os
import pickle

import torch
import time
import numpy as np

from torch.autograd import Variable

import py_ner.ner_data_loader as loader
import  py_ner.lstm_cnn_crf_utils as utils

t = time.time()

# python -m visdom.server

optparser = optparse.OptionParser()
optparser.add_option(
    "-t", "--test", default="./eng.testb",
    help="Test set location"
)
optparser.add_option(
    '--score', default='./temp/score.txt',
    help='score file location'
)
optparser.add_option(
    "-f", "--crf", default="0",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    '--loss', default='loss.txt',
    help='loss file location'
)
optparser.add_option(
    '--model_path', default='models/lstm_cnn_crf',
    help='model path'
)
optparser.add_option(
    '--map_path', default='models/mapping.pkl',
    help='model path'
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)

opts = optparser.parse_args()[0]

mapping_file = opts.map_path

with open(mapping_file, 'rb') as f:
    mappings = pickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

use_gpu =  opts.use_gpu == 1 and torch.cuda.is_available()


assert os.path.isfile(opts.test)
assert parameters['tag_scheme'] in ['iob', 'iobes']

if not os.path.isfile(utils.eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % utils.eval_script)
if not os.path.exists(utils.eval_temp):
    os.makedirs(utils.eval_temp)

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

test_sentences = loader.load_sentences(opts.test, lower, zeros)
loader.update_tag_scheme(test_sentences, tag_scheme)
test_data = loader.prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

model = torch.load(opts.model_path)
model_name = opts.model_path.split('/')[-1].split('.')[0]

if use_gpu:
    model.cuda()

model.eval()

# def getmaxlen(tags):
#     l = 1
#     maxl = 1
#     for tag in tags:
#         tag = id_to_tag[tag]
#         if 'I-' in tag:
#             l += 1
#         elif 'B-' in tag:
#             l = 1
#         elif 'E-' in tag:
#             l += 1
#             maxl = max(maxl, l)
#             l = 1
#         elif 'O' in tag:
#             l = 1
#     return maxl

def eval(model, datas, maxl=1):
    prediction = []
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        # l = getmaxlen(ground_truth_id)
        # if not l == maxl:
        #     continue
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(),chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    predf = utils.eval_temp + '/pred.' + model_name
    scoref = utils.eval_temp + '/score.' + model_name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (utils.eval_script, predf, scoref))

    with open(scoref, 'rb') as f:
        for l in f.readlines():
            print(l.strip())

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))

eval(model, test_data, 200)
# for l in range(1, 6):
#     print('maxl=', l)
#     eval(model, test_data, l)
#     # print()
# # for i in range(10):
# #     eval(model, test_data, 100)

print(time.time() - t)


'''
model_testing_sentences = ['Jay is from India', 'Donald is the president of USA']

# parameters
lower = parameters['lower']

# preprocessing
final_test_data = []
for sentence in model_testing_sentences:
    s = sentence.split()
    str_words = [w for w in s]
    words = [word_to_id[utils.lower_case(w, lower) if utils.lower_case(w, lower) in word_to_id else '<UNK>'] for w in str_words]

    # Skip characters that are not in the training set
    chars = [[char_to_id[c] for c in w if c in char_to_id] for w in str_words]

    final_test_data.append({
        'str_words': str_words,
        'words': words,
        'chars': chars,
    })

# prediction
predictions = []
print("Prediction:")
print("word : tag")
for data in final_test_data:
    words = data['str_words']
    chars2 = data['chars']

    d = {}

    # Padding the each word to max word size of that sentence
    chars2_length = [len(c) for c in chars2]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
    for i, c in enumerate(chars2):
        chars2_mask[i, :chars2_length[i]] = c
    chars2_mask = Variable(torch.LongTensor(chars2_mask))

    dwords = Variable(torch.LongTensor(data['words']))

    # We are getting the predicted output from our model
    if use_gpu:
        val, predicted_id = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
    else:
        val, predicted_id = model(dwords, chars2_mask, chars2_length, d)

    pred_chunks = utils.get_chunks(predicted_id, tag_to_id)
    temp_list_tags = ['NA'] * len(words)
    for p in pred_chunks:
        temp_list_tags[p[1]] = p[0]

    for word, tag in zip(words, temp_list_tags):
        print(word, ':', tag)
    print('\n')
'''