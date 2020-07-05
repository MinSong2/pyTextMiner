from gensim.models import FastText
import pyTextMiner as ptm

fname ='korean_sns_comments_ft.bin'

#existent_word = "싸기꾼"
existent_word = "사기꾼"

mode = 'jamo_split'
if mode != 'jamo_split':
    model = FastText.load(fname)

    if existent_word in model.wv.vocab:
        print('True')
    else:
        print('False')

    computer_vec = model.wv[existent_word]  # numpy vector of a word
    print(str(computer_vec))

    similarities = model.wv.most_similar(existent_word)
    print(str(similarities))

else:
    util = ptm.Utility()

    fname = 'korean_sns_comments_jamo_ft.bin'
    model = FastText.load(fname)

    jamo_existent_word = util.jamo_sentence(existent_word)
    if jamo_existent_word in model.wv.vocab:
        print('True')
    else:
        print('False')

    computer_vec = model.wv[jamo_existent_word]  # numpy vector of a word
    print(str(computer_vec))

    similarities = util.most_similar(existent_word,model)
    print(str(similarities))
