import pycrfsuite

def read_file(file_name):
    sents = []
    with open(file_name,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for idx,l in enumerate(lines) :
            if l[0]==';' and lines[idx+1][0]=='$':
                this_sent = []
            elif l[0]=='$' and lines[idx-1][0]==';':
                continue
            elif l[0]=='\n':
                sents.append(this_sent)
            else :
                this_sent.append(tuple(l.split()))
    return sents

train_sents = read_file('data/train.txt')
test_sents = read_file('data/test.txt')

def word2features(sent, i):
    last_p = sent[i][0]
    mor_idx = sent[i][1]
    word = sent[i][3]
    postag = sent[i][4]
    features = [
        'bias',
        'word=' + word,
        'word[:1]=' + word[:1],
        'word[:2]=' + word[:2],
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'last_p=' + last_p,
        'mor_i=' + mor_idx,
        'len='+str(len(word))
    ]
    if i > 0:
        word1 = sent[i-1][3]
        postag1 = sent[i-1][4]
        features.extend([
            '-1:word=' + word1,
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i+1][3]
        postag1 = sent[i+1][4]
        features.extend([
            '+1:word=' + word1,
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features

def add_features_to_sent(sent):
    new_sent = []
    this_ej = []
    ej_idx = 1
    morph_idx = 1
    for w in sent:
        if int(w[0]) > ej_idx:
            assert(len(this_ej)>0)
            last_particle = '-'
            if this_ej[-1][3][0]=='J':
                last_particle=this_ej[-1][2]+'/'+this_ej[-1][3]
            this_ej = [(last_particle,)+m for m in this_ej]
            new_sent += this_ej
            this_ej = []
            ej_idx += 1
            morph_idx = 1
        this_ej.append((str(morph_idx),)+w)
        morph_idx += 1
    if len(this_ej) > 0:
        last_particle = '-'
        if this_ej[-1][3][0] == 'J':
            last_particle = this_ej[-1][2] + '/' + this_ej[-1][3]
        this_ej = [(last_particle,) + m for m in this_ej]
        new_sent += this_ej
    return new_sent

def sent2features(sent):
    sent = add_features_to_sent(sent)
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for ej_idx, token, postag, label in sent]


def sent2tokens(sent):
    return [token for ej_idx, token, postag, label in sent]

#data에서 자질 추출
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

#중간 출력
print(X_train[0][2])

#모델 학습
trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('model/kor.crfsuite')
#모델 로드
tagger = pycrfsuite.Tagger()
tagger.open('model/kor.crfsuite')
def make_tag_idx(ans_seq):
    B_num = 0
    all_answer_start = []
    all_answer_end = []
    all_answer_tag = []

    for sent in ans_seq:
        start_idx = []
        end_idx = []
        tag_set = []
        tag_num = 0
        flag = 0

        for tag in sent:
            try:
                if flag == 1 and tag[0] != 'I':
                    end_idx.append(tag_num - 1)
                    flag = 0
            except:
                print()

            if tag[0] == 'B':
                B_num = B_num + 1
                start_idx.append(tag_num)
                tag_set.append(tag[2:4])
                flag = 1
            tag_num = tag_num + 1

        if flag == 1:
            end_idx.append(tag_num - 1)

        all_answer_start.append(start_idx)
        all_answer_end.append(end_idx)
        all_answer_tag.append(tag_set)

    return (all_answer_start, all_answer_end, all_answer_tag, B_num)

def eval(pred_seq, ans_seq):  # changed

    TP = 0

    (all_answer_start, all_answer_end, all_answer_tag, answer_num) = make_tag_idx(ans_seq)
    (all_pred_start, all_pred_end, all_pred_tag, pred_num) = make_tag_idx(pred_seq)

    for i in range(0, len(ans_seq)):
        for j in range(0, len(all_pred_start[i])):
            for k in range(0, len(all_answer_start[i])):
                if all_pred_start[i][j] == all_answer_start[i][k] and all_pred_end[i][j] == all_answer_end[i][k] and \
                                all_pred_tag[i][j] == all_answer_tag[i][k]: TP = TP + 1

    return (TP, pred_num, answer_num)

def write_prediction(file_name, sents, preds):
    with open(file_name,'w', encoding='utf-8') as f:
        for s, s_p in zip(sents,preds):
            f.write(";\n$\n")
            for t, p in zip(s,s_p) :
                cmp = '' if t[2]==p else '*'
                f.write(t[0]+' '+t[1]+' cor:'+t[2]+' pred:'+p+cmp+'\n')
            f.write('\n')
#모델 평가

example_sent = test_sents[0]
print(' '.join(sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))

y_pred = [tagger.tag(xseq) for xseq in X_test]
write_prediction('../data/conll2002_esp.pred', test_sents, y_pred)
TP, pred_num, answer_num = eval(y_test, y_pred)
print(TP, pred_num, answer_num)

#Precision, Recall, F1score

precision = float(TP) / pred_num
recall = float(TP) / answer_num
f1score = 2.0 * (precision * recall) / (precision + recall)
print('\n\nprecision : '+str(precision*100)+'%\n'+'recall : '+str(recall*100)+'%\n'+'f1 score : '+str(f1score*100)+'%\n')