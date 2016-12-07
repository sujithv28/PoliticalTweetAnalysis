#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import csv
import operator
import re
import string
import pandas as pd
import numpy.lib
import numpy as np
import pdb
import cPickle as pickle
import vincent
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Helper methods which tokenize, and convert the content string
# to a list of words (can also handle #'s, @'s, etc)
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

positive_vocab = [
    'good', 'nice', 'great', 'awesome', 'outstanding',
    'fantastic', 'terrific', ':)', ':-)', 'like', 'love',
]
negative_vocab = [
    'bad', 'terrible', 'evil', 'useless', 'hate', ':(', ':-(',
    'scandal', 'racist'
]

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', '...', 'I']
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

class App:
    def __init__(self, file_name):
        self.csv_f = csv.reader(file_name)
        self.twitter_data = []
        self.corpus = []
        self.data = None
        self.headers = None
        self.terms_filtered = []
        self.terms_all = []
        self.terms_trump = []
        self.terms_hillary = []
        self.terms_all_counter = Counter()
        self.terms_filtered_counter = Counter()
        self.tfidf_matrix = None
        self.fname = 'tweets_dataframe'

    def load_data(self):
        i = 0
        for row in self.csv_f:
            # Append every row after header to list
            if i>0:
                if (i%1000 == 0):
                    print('Analyzed %10d' % (i))
                # Filter out "RT" text if it's a retweet
                if len(row[5])>2 and row[5][:2] == 'RT':
                    row[5] = row[5][3:]

                unicode_tweet = unicode(row[5], errors='replace')
                self.corpus.append(unicode_tweet)

                row[5] = [term for term in preprocess(row[5]) if term not in stop]
                self.terms_filtered.extend([term for term in row[5] if not term.startswith(('#', '@'))])
                self.terms_all.extend(row[5])

                if 'hillary' in self.terms_all or 'clinton' in self.terms_all:
                    self.terms_hillary.extend(row[5])
                if 'trump' in self.terms_all or 'donald' in self.terms_all:
                    self.terms_trump.extend(row[5])

                self.twitter_data.append(row)
            else:
                # Save Column Headers (First row of CSV File)
                self.headers = row
            i+=1

            # Construct Pandas Data Frame from List
            self.data = pd.DataFrame(self.twitter_data, columns=self.headers)
            # Count terms only once, equivalent to Document Frequency
            self.terms_single = set(self.terms_filtered)


    def create_counters(self):
        # Print out 10 most frequent words filtered
        print('\nFiltered Frequency:')
        self.terms_filtered_counter = Counter(self.terms_filtered)
        for word, count in self.terms_filtered_counter.most_common(15):
            print('{0}: {1}'.format(word, count))

        # Print out 10 most unfiltered frequent words
        print('\nUnfiltered Frequency:')
        self.terms_all_counter = Counter(self.terms_filtered)
        for word, count in self.terms_all_counter.most_common(15):
            print('{0}: {1}'.format(word, count))

    def create_tfidf(self):
        # Print and generate tfidf matrix from all the tweets
        print('\nTf-Idf Vectors:')
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
        self.tfidf_matrix =  tf.fit_transform(self.corpus)
        feature_names = tf.get_feature_names()
        dense = self.tfidf_matrix.todense()
        densetweets = dense[0].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(densetweets)), densetweets) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
           print('{0: <20} {1}'.format(phrase, score))

    def save_dataframe(self):
        np.save(open(self.fname, 'w'), self.data)
        if len(self.data.shape) == 2:
            meta = self.data.index,self.data.columns
        elif len(self.data.shape) == 1:
            meta = (self.data.index,)
        else:
            raise ValueError('save_pandas: Cannot save this type')
        s = pickle.dumps(meta)
        s = s.encode('string_escape')
        with open(self.fname, 'a') as f:
            f.seek(0, 2)
            f.write(s)

    def load_dataframe(self):
        values = np.load(self.fname, mmap_mode=mmap_mode)
        with open(self.fname) as f:
            numpy.lib.format.read_magic(f)
            numpy.lib.format.read_array_header_1_0(f)
            f.seek(values.dtype.alignment*values.size, 1)
            meta = pickle.loads(f.readline().decode('string_escape'))
        if len(meta) == 2:
            return pd.DataFrame(values, index=meta[0], columns=meta[1])
        elif len(meta) == 1:
            return pd.Series(values, index=meta[0])

def main():
    import sys
    temp_subset_file = open('/Users/sujith/Dropbox/US_UK_ElectionTweets/US_all_tweets/temp_subset.csv')
    all_tweets_file = open('/Users/sujith/Dropbox/US_UK_ElectionTweets/US_all_tweets/all_tweets.csv')
    app = App(temp_subset_file)
    # app.load_dataframe()
    app.load_data()
    app.create_counters()
    app.create_tfidf()
    app.save_dataframe()
    pdb.set_trace()

if __name__ == '__main__':
    main()
    print('\n')
