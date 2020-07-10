import pyTextMiner as ptm

corpus=ptm.CorpusFromFile('./data/2016-10-20.txt')
pmi=ptm.pmi.PMICalculator(corpus)
sent='아이오아이'
result=pmi.__call__(sent)
print(result)