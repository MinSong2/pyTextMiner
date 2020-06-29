
import sys
import re
import pickle
import numpy as np

CLASS_1 = 1  # next is space
CLASS_0 = 0  # next is not space

S1 = re.compile(u'''[\s]+''')


def snorm(string):
    return S1.sub(' ', string.replace('\t', ' ')).strip()


def open_file(filename, mode):
    try:
        fid = open(filename, mode)
    except:
        sys.stderr.write("open_file(), file open error : %s\n" % (filename))
        exit(1)
    else:
        return fid


def close_file(fid):
    fid.close()


def build_dictionary(train_path, padd):
    char_rdic = []
    visit = {}
    fid = open_file(train_path, 'r')
    for line in fid:
        line = line.strip()
        if line == "": continue
        for c in line:
            if c not in visit:
                char_rdic.append(c)
                visit[c] = 1
    if padd not in visit: char_rdic.append(padd)
    char_dic = {w: i for i, w in enumerate(char_rdic)}  # char to id
    close_file(fid)
    return char_dic


def one_hot(i, size):
    return [1 if j == i else 0 for j in range(size)]


def get_xy_data(sentence, pos, n_steps, padd):
    slen = len(sentence)
    x_data = []
    y_data = []
    next_pos = -1
    count = 0
    i = pos
    while i < slen:
        c = sentence[i]
        x_data.append(c)
        next_c = None
        if i + 1 < slen: next_c = sentence[i + 1]
        if next_c == ' ':
            y_data.append(CLASS_1)
        else:
            y_data.append(CLASS_0)
        count += 1
        i += 1
        if count == n_steps: break
    if count == n_steps:
        if i == slen:
            # reached end
            next_pos = -1
        if i < slen:
            # move prev space + 1
            j = i - 1
            space_count = 0
            while j > 0:
                c = sentence[j]
                if c == ' ':
                    space_count += 1
                    if space_count == 1: break
                if i - j >= 12:  # no prev space atmost 12
                    break
                j -= 1
            if j <= i - 1:
                next_pos = j + 1
    else:
        # padding
        diff = n_steps - count
        x_data += [padd] * diff
        y_data += [CLASS_0] * diff
        next_pos = -1
    return x_data, y_data, next_pos, count


def next_batch(sentence, pos, char_dic, vocab_size, n_steps, padd):
    '''
    y_data =  1 or 0     => n_steps unfolding => [0,0,1,0,...]
    ^
    |
    x_data = [1,0,...,0] => n_steps unfolding => [[1,0,..0],..,[0,0,1,..0]]

    batch_xs.shape => (batch_size=1, n_steps, n_input)
    batch_ys.shape => (batch_size=1, n_steps)
    '''
    batch_xs = []
    batch_ys = []
    x_data, y_data, next_pos, count = get_xy_data(sentence, pos, n_steps, padd)
    x_data = [char_dic[c] if c in char_dic else padd for c in x_data]
    x_data = [one_hot(i, vocab_size) for i in x_data]
    batch_xs.append(x_data)
    batch_ys.append(y_data)
    batch_xs = np.array(batch_xs, dtype='f')
    batch_ys = np.array(batch_ys, dtype='int32')
    return batch_xs, batch_ys, next_pos, count


def test_next_batch(train_path, char_dic, vocab_size, n_steps, padd):
    fid = open_file(train_path, 'r')
    for line in fid:
        line = line.strip()
        if line == "": continue
        line = line.decode('utf-8')
        sentence = snorm(line)
        pos = 0
        while pos != -1:
            batch_xs, batch_ys, next_pos, count = next_batch(sentence, pos, char_dic, vocab_size, n_steps, padd)
            print
            'window : ' + sentence[pos:pos + n_steps].encode('utf-8')
            print
            'count : ' + str(count)
            print
            'next_pos : ' + str(next_pos)
            print
            batch_ys
            pos = next_pos
    close_file(fid)


def get_validation_data(validation_path, char_dic, vocab_size, n_steps, padd):
    validation_data = []
    fid = open_file(validation_path, 'r')
    for line in fid:
        line = line.strip()
        if line == "": continue
        sentence = snorm(line)
        pos = 0
        while pos != -1:
            batch_xs, batch_ys, next_pos, count = next_batch(sentence, pos, char_dic, vocab_size, n_steps, padd)
            validation_data.append((batch_xs, batch_ys, count))
            pos = next_pos
    close_file(fid)
    return validation_data


def to_sentence(tag_vector, sentence):
    out = []
    j = 0
    tag_vector_size = len(tag_vector)
    sentence_size = len(sentence)
    while j < tag_vector_size and j < sentence_size:
        tag = tag_vector[j]
        if tag == CLASS_1:
            out.append(sentence[j])
            if sentence[j] != ' ': out.append(' ')
        else:
            out.append(sentence[j])
        j += 1
    n_sentence = ''.join(out)
    return snorm(n_sentence)


# -------------------------------------------------------------------------

def build_dictionary_emb(embedding_dir):
    embedding_path = embedding_dir + '/' + 'embedding.pickle'
    with open(embedding_path, 'rb') as handle:
        embeddings = pickle.load(handle)
    id2ch = {}
    vocab_path = embedding_dir + '/' + 'vocab.txt'
    idx = 0
    with open(vocab_path, "r") as handle:
        for line in handle:
            ch, count = line.split(' ')
            id2ch[idx] = ch.decode('utf-8')
            idx += 1
    assert (idx == len(embeddings))
    ch2id = {}
    id2emb = {}
    for i, ch in id2ch.iteritems():
        ch2id[ch] = i
        try:
            id2emb[i] = embeddings[i]
        except Exception as e:
            sys.stderr.write('vocab.txt, embedding.pickle not aligned\n')
            sys.exit(1)
    embedding_dim = len(embeddings[0])
    return ch2id, id2ch, id2emb, embedding_dim


def next_batch_emb(sentence, pos, char_dic, id2emb, n_steps, padd):
    '''
    y_data =  1 or 0     => n_steps unfolding => [0,0,1,0,...]
    ^
    |
    x_data = [1,0,...,0] => n_steps unfolding => [[1.0,1.4,..3.2],..,[0.2,0.6,3.7,..,0.0]]

    batch_xs.shape => (batch_size=1, n_steps, n_input)
    batch_ys.shape => (batch_size=1, n_steps)
    '''
    batch_xs = []
    batch_ys = []
    x_data, y_data, next_pos, count = get_xy_data(sentence, pos, n_steps, padd)
    tmp_x_data = []
    for c in x_data:
        id = 0  # for 'UNK'
        if c in char_dic: id = char_dic[c]
        '''
        print 'c, id = ', c.encode('utf-8'), id
        print id2emb[id]
        '''
        tmp_x_data.append(id2emb[id])
    x_data = tmp_x_data
    batch_xs.append(x_data)
    batch_ys.append(y_data)
    batch_xs = np.array(batch_xs, dtype='f')
    batch_ys = np.array(batch_ys, dtype='int32')
    return batch_xs, batch_ys, next_pos, count


def test_next_batch_emb(train_path, char_dic, id2emb, n_steps, padd):
    fid = open_file(train_path, 'r')
    for line in fid:
        line = line.strip()
        if line == "": continue
        line = line.decode('utf-8')
        sentence = snorm(line)
        pos = 0
        while pos != -1:
            batch_xs, batch_ys, next_pos, count = next_batch_emb(sentence, pos, char_dic, id2emb, n_steps, padd)
            print
            'window : ' + sentence[pos:pos + n_steps].encode('utf-8')
            print
            'count : ' + str(count)
            print
            'next_pos : ' + str(next_pos)
            print
            batch_xs
            print
            batch_ys
            pos = next_pos
    close_file(fid)


def get_validation_data_emb(validation_path, char_dic, id2emb, n_steps, padd):
    validation_data = []
    fid = open_file(validation_path, 'r')
    for line in fid:
        line = line.strip()
        if line == "": continue
        line = line.decode('utf-8')
        sentence = snorm(line)
        pos = 0
        while pos != -1:
            batch_xs, batch_ys, next_pos, count = next_batch_emb(sentence, pos, char_dic, id2emb, n_steps, padd)
            validation_data.append((batch_xs, batch_ys, count))
            pos = next_pos
    close_file(fid)
    return validation_data