from gensim import corpora, models
from collections import defaultdict
import jieba
import pickle

class NLModel:

    def __init__(self):
        self.stoplist = list()
        self.dictionary = corpora.Dictionary()

    def loadStopwords(self):
        lines = []
        with open('data/stopwords.txt') as f:
            lines = f.readlines()
        for line in lines:
            self.stoplist.append(line.strip().decode('utf-8'))
        print '<stop words loaded, total record ' + str(len(self.stoplist)) + '>'

    def splitTexts(self, documents, fname = ''):

    # split chinese text using jieba

        documents = [jieba.cut(text) for text in documents]
        documents = [[token for token in text if token not in self.stoplist] for text in documents]
        if fname != '':
            with open('save/'+fname+'.pk', 'wb') as f:
                pickle.dump(documents, f)
        '''
        texts = [[token for token in text if token not in self.stoplist] for text in documents]
        freq = defaultdict(int)
        for text in texts:
            for token in text:
                freq[token] += 1
        texts = [[token for token in text if freq[token]>1] for text in texts]
        '''
        return documents

    def updateDict(self, documents):
        self.dictionary.add_documents(documents, prune_at=None)
        print "<dictionary updated, total record %d>"%len(self.dictionary.keys())

    def filterDict(self, l, u=1.0):

    # filter most frequent tokens and least frequent ones
    # l is the lower threshold
    # u is the upper threshold
    # this is a wrap for gensim.corpora.Dictionary.filter_extremes

        self.dictionary.filter_extremes(no_below=l, no_above=u, keep_n=None, keep_tokens=None)
        print "<dictionary updated, total record %d>"%len(self.dictionary.keys())

    def computeCorpus(self, documents):

    # translate tokens into (id, freq) pairs, namely bag-of-word

        corpus = [self.dictionary.doc2bow(text) for text in documents]
        return corpus

    def translate(self, token_id):
        return self.dictionary[token_id]

    @staticmethod
    def computeTfidf(corpus, context = None, normalize = True):
        if context == None:
            context = corpus
        tfidf_model = models.TfidfModel(context, normalize=normalize)
        tfidf = tfidf_model[corpus]
        return tfidf, tfidf_model

    @staticmethod
    def getFrequentTerm(bows, k, reverse = False):

    # get top k most frequent tokens
    # bows is a list of bag-of-word
    # [[(token, freq), ...], ...]

        freq_dict = defaultdict(int)
        for bow in bows:
            for term_id, term_freq in bow:
                freq_dict[term_id] = term_freq
        lst = []
        for key in freq_dict:
            val = freq_dict[key]
            lst.append((key, val))
        lst.sort(key=lambda x:x[1], reverse=not reverse)
        return lst[:k]

    def extractLSI(self, corpus, num_topics):
        lsi = models.lsimodel.LsiModel(corpus =      corpus,
                                       id2word =     self.dictionary,
                                       num_topics =  num_topics)
        return lsi

    def extractLDA(self, corpus, num_topics, chunksize):
        lda = models.ldamodel.LdaModel(corpus =          corpus,
                                       id2word =         self.dictionary,
                                       num_topics =      num_topics,
                                       update_every =    1,
                                       chunksize =       10,
                                       passes =          1)
        return lda
