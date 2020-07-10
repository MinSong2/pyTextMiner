# -*- encoding:utf8 -*-
import pyTextMiner as ptm

dictionary_path='./dict/user_dic.txt'

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Komoran(),
                        #ptm.tokenizer.WordPos(),
                        ptm.helper.POSFilter('NN*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'),
                        ptm.counter.WordCounter())

corpus = ptm.CorpusFromFile('./data/sampleEng.txt')

#corpus = ptm.CorpusFromFile('Gulliver_Travels.txt')
#pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
#                        ptm.tokenizer.Word(),
#                        ptm.counter.WordCounter())
result = pipeline.processCorpus(corpus)

print(result)
print()

doc_collection = ''
term_counts = {}
for doc in result:
    for sent in doc:
        for _str in sent:
            term_counts[_str[0]] = term_counts.get(_str[0], 0) + int(_str[1])
            freq = range(int(_str[1]))
            co = ''
            for n in freq:
                co +=  ' ' + _str[0]

            doc_collection += ' ' + co
word_freq = []
for key, value in term_counts.items():
    word_freq.append((value,key))

word_freq.sort(reverse=True)
print(word_freq)

f = open("demo_result.txt", "w", encoding='utf8')
for pair in word_freq:
    f.write(pair[1] + '\t' + str(pair[0]) + '\n')
f.close()

from wordcloud import WordCloud

# Read the whole text.

# Generate a word cloud image
wordcloud = WordCloud().generate(doc_collection)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt

# Window의 경우 폰트 경로
# font_path = 'C:/Windows/Fonts/malgun.ttf'

#for Mac
#font_path='/Library/Fonts/AppleGothic.ttf'

# lower max_font_size
wordcloud = WordCloud(max_font_size=40,
                      background_color='white',
                      collocations=False)

wordcloud.generate(doc_collection)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
