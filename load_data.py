#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import argparse
import vincent
import os, pwd
import HTMLParser
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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
        self.file_name = file_name
        self.twitter_data = []
        self.corpus = []
        self.df = None
        self.headers = None
        self.terms_filtered = []
        self.terms_all = []
        self.terms_trump = []
        self.terms_hillary = []
        self.geo_data = []
        self.terms_all_counter = Counter()
        self.terms_filtered_counter = Counter()
        self.tfidf_matrix = None
        self.fname = 'tweets_dataframe.pickle'

    def tweet_to_list(self, str):
        # Filter out "RT" text if it's a retweet
        if len(str)>2 and str[:2] == 'RT':
            str = str[3:]
        list = [term for term in preprocess(str) if term not in stop]
        return list

    def geostamp_to_list(self, str):
        if (str != ''):
            locations_str = str.replace('[','').split('],')
            lists = [map(float, s.replace(']','').split(',')) for s in locations_str]
            list = lists
        else:
            list = []
        return list

    def str_to_bool(self, val):
        if (val == 1):
            return True
        else:
            return False

    def load_data(self, load=False):
        i = 0
        if (load or not(os.path.exists(self.fname))):
            print("Creating Data Frame from Scratch")
            new_df = pd.read_csv(self.file_name)
            new_df['Content'] = new_df.apply(lambda row: self.tweet_to_list(row['Content']), axis=1)
            new_df['Geostamp'] = new_df.apply(lambda row: self.geostamp_to_list(row['Geostamp']), axis=1)
            new_df['isHillary'] = new_df.apply(lambda row: self.str_to_bool(row['isHillary']), axis=1)
            new_df['isTrump'] = new_df.apply(lambda row: self.str_to_bool(row['isTrump']), axis=1)
            self.df = new_df
        else:
            self.df = pd.read_pickle(self.fname)
        print(list(self.df.columns.values))
        print(self.df)

    def analyze_data(self):
        for index, row in self.df.iterrows():
            str = ' '.join(row['Content'])
            unicode_tweet = unicode(str, errors='replace')
            self.corpus.append(unicode_tweet)
            
            self.terms_all.extend(row['Content'])
            filtered_list = [term for term in row['Content'] if not term.startswith(('#', '@'))]
            self.terms_filtered.extend(filtered_list)

            if(row['isHillary']):
                self.terms_hillary.extend(row['Content'])
            else:
                self.terms_hillary.extend(row['Content'])
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
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:len(sorted_phrase_scores)]:
           print('{0: <40} {1}'.format(phrase, score))

    def find_cosine_similar(self, tfidf_matrix, index, top_n = 5):
        cosine_similarities = linear_kernel(self.tfidf_matrix[index:index+1], self.tfidf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
        return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    def print_cosine_similar(self):
        print('\nTweets Similar To: %s \n' % (self.corpus[15]))
        for index, score in self.find_cosine_similar(self.tfidf_matrix, 15):
            print('%.2f  ->  %s' % (score, self.corpus[index]))

def main():
    import sys

    username = pwd.getpwuid( os.getuid() )[ 0 ]

    temp_subset_file = open('/Users/{0:s}/Dropbox/US_UK_ElectionTweets/US_all_tweets/temp_subset.csv'.format(username))
    all_tweets_file = open('/Users/{0:s}/Dropbox/US_UK_ElectionTweets/US_all_tweets/all_tweets.csv'.format(username))
    geo_tweets_file = open('/Users/{0:s}/Dropbox/US_UK_ElectionTweets/geo_time_tweets_fixed/fixed_geo.csv'.format(username))
    temp_geo_tweets_file = open('/Users/{0:s}/Dropbox/US_UK_ElectionTweets/geo_time_tweets_fixed/temp_geo.csv'.format(username))
    app = App(temp_geo_tweets_file)

    parser = argparse.ArgumentParser(description='Analyze Political Twitter Data')
    parser.add_argument("-l", "--load", action="store_true", required=False,
                        help="Loads a CSV file from scratch rather than using existing Dataframe.")
    args = parser.parse_args()

    app.load_data(args.load)
    app.analyze_data()
    app.create_counters()
    app.create_tfidf()
    app.print_cosine_similar()
    app.df.to_pickle(app.fname)

    print('\n')
    # pdb.set_trace()

if __name__ == '__main__':
    main()
