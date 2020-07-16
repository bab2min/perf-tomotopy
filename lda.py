import tomotopy as tp
import nltk
import re
from sklearn.datasets import fetch_20newsgroups
import utils

with utils.Timer('LDAModel.__init__'):
    mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=8, rm_top=10, k=40)

porter_stemmer = nltk.PorterStemmer().stem
pat = re.compile('^[a-z]{2,}$')
newsgroups_train = fetch_20newsgroups()
with utils.Timer('LDAModel.add_doc'):
    for d in newsgroups_train.data:
        mdl.add_doc(filter(lambda x:pat.match(x), (porter_stemmer(w) for w in d.lower().split())))

with utils.Timer('LDAModel.train'):
    for i in range(0, 1000, 20):
        print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))
        mdl.train(20)
    print('Iteration: {:04}, LL per word: {:.4}'.format(1000, mdl.ll_per_word))

utils.Timer.print()
