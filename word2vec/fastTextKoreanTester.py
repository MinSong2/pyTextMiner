from gensim.models import FastText
fname ='../embeddings/fastText_ko/korean_sns_comments_ft.bin'
model = FastText.load(fname)

existent_word = "사기꾼"
if existent_word in model.wv.vocab:
    print('True')
else:
    print('False')

computer_vec = model.wv[existent_word]  # numpy vector of a word
print(str(computer_vec))

similarities = model.wv.most_similar(existent_word)
print(str(similarities))