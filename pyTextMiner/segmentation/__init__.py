from chatspace import ChatSpace
from pycrfsuite_spacing import PyCRFSuiteSpacing
from pycrfsuite_spacing import TemplateGenerator
from pycrfsuite_spacing import CharacterFeatureTransformer

class BaseSegmentation:
    IN_TYPE = [str]
    OUT_TYPE = [str]

class SegmentationKorean(BaseSegmentation):
    def __init__(self, model=None):
        #model_path = 'demo_model.crfsuite'
        to_feature = CharacterFeatureTransformer(
            TemplateGenerator(
                begin=-2,
                end=2,
                min_range_length=3,
                max_range_length=3)
        )
        self.inst = PyCRFSuiteSpacing(to_feature)
        self.inst.load_tagger(model)

    def __call__(self, *args, **kwargs):
        return self.inst(args[0])

class ChatSpaceSegmentationKorean(BaseSegmentation):
    def __init__(self):
        self.inst = ChatSpace()

    def __call__(self, *args, **kwargs):
        return self.inst.space(args[0], batch_size=64)

import sys
import os
import re
import pickle
import numpy as np
import tensorflow as tf
import pyTextMiner.segmentation.lstmWordSegmentationModel as model
import pyTextMiner.segmentation.wordSegmentationModelUtil as util

class LSTMSegmentationKorean(BaseSegmentation):
    def __init__(self, model_path='./model'):
        self.model = model_path
        dic_path = self.model + '/' + 'dic.pickle'

        # config
        self.n_steps = 30  # time steps
        self.padd = '\t'  # special padding chracter
        with open(dic_path, 'rb') as handle:
            self.char_dic = pickle.load(handle)  # load dic
        n_input = len(self.char_dic)  # input dimension, vocab size
        self.n_hidden = 8  # hidden layer size
        self.n_classes = 2  # output classes,  space or not
        self.vocab_size = n_input

        self.x = tf.placeholder(tf.float32, [None, self.n_steps, n_input])
        self.y_ = tf.placeholder(tf.int32, [None, self.n_steps])
        self.early_stop = tf.placeholder(tf.int32)

        # LSTM layer
        # 2 x n_hidden = state_size = (hidden state + cell state)
        self.istate = tf.placeholder(tf.float32, [None, 2 * self.n_hidden])
        weights = {
            'hidden': model.weight_variable([n_input, self.n_hidden]),
            'out': model.weight_variable([self.n_hidden, self.n_classes])
        }
        biases = {
            'hidden': model.bias_variable([self.n_hidden]),
            'out': model.bias_variable([self.n_classes])
        }

        self.y = model.RNN(self.x, self.istate, weights, biases, self.n_hidden, self.n_steps, n_input, self.early_stop)

        self.batch_size = 1
        self.logits = tf.reshape(tf.concat(self.y, 1), [-1, self.n_classes])

        NUM_THREADS = 1
        config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                                inter_op_parallelism_threads=NUM_THREADS,
                                log_device_placement=False)
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()  # save all variables
        checkpoint_dir = self.model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            sys.stderr.write("model restored from %s\n" % (ckpt.model_checkpoint_path))
        else:
            sys.stderr.write("no checkpoint found" + '\n')
            sys.exit(-1)

    def __call__(self, *args, **kwargs):
        sentence = args[0]
        sentence_size = len(sentence)
        tag_vector = [-1] * (sentence_size + self.n_steps)  # buffer n_steps
        pos = 0
        while pos != -1:
            batch_xs, batch_ys, next_pos, count = util.next_batch(sentence, pos, self.char_dic, self.vocab_size, self.n_steps, self.padd)
            '''
            print 'window : ' + sentence[pos:pos+n_steps]
            print 'count : ' + str(count)
            print 'next_pos : ' + str(next_pos)
            print batch_ys
            '''
            c_istate = np.zeros((self.batch_size, 2 * self.n_hidden))
            feed = {self.x: batch_xs, self.y_: batch_ys, self.istate: c_istate, self.early_stop: count}
            argmax = tf.arg_max(self.logits, 1)
            result = self.sess.run(argmax, feed_dict=feed)

            # overlapped copy and merge
            j = 0
            result_size = len(result)
            while j < result_size:
                tag = result[j]
                if tag_vector[pos + j] == -1:
                    tag_vector[pos + j] = tag
                else:
                    if tag_vector[pos + j] == util.CLASS_1:  # 1
                        if tag == util.CLASS_0:  # 1 -> 0
                            sys.stderr.write("1->0\n")
                            tag_vector[pos + j] = tag
                    else:  # 0
                        if tag == util.CLASS_1:  # 0 -> 1
                            sys.stderr.write("0->1\n")
                            tag_vector[pos + j] = tag
                j += 1
            pos = next_pos
        # generate output using tag_vector
        print
        'out = ' + util.to_sentence(tag_vector, sentence)

        return util.to_sentence(tag_vector, sentence)


    def close(self):
        self.sess.close()