#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

import argparse
import json
import os
import pwd
import re
import string
from collections import Counter
from datetime import datetime

import pandas as pd
from nltk.corpus import stopwords
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
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
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
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def extract_http_link(s):
    r = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(r, s)
    if match:
        return match.group()
    return ''


class App:
    def __init__(self, file_name):
        self.file_name = file_name
        self.twitter_data = []
        self.corpus = []
        self.df = None
        self.headers = None
        self.terms = {
            'all': [],
            'filtered': [],
            'hillary': [],
            'trump': [],
            'positive': [],
            'neutral': [],
            'negative': []
        }
        self.geo_data = []
        self.terms_all_counter = Counter()
        self.terms_filtered_counter = Counter()
        self.tfidf_matrix = None
        self.fname = 'tweets_dataframe.pickle'
        self.geo_data = {
            'type': 'FeatureCollection',
            'features': []
        }

    def tweet_to_list(self, tweet):
        # Filter out 'RT' text if it's a retweet
        if len(tweet) > 2 and tweet[:2] == 'RT':
            tweet = tweet[3:]
        l = [term for term in preprocess(tweet) if term not in stop]
        l = [item.lower() for item in l]
        return l

    def geostamp_to_list(self, geostamp_str):
        list = []
        try:
            if (geostamp_str != ''):
                locations_str = geostamp_str.replace('[', '').split('],')
                lists = [map(float, s.replace(']', '').split(',')) for s in locations_str]
                list = lists
        except ValueError, e:
            print('Error: %s' % (geostamp_str))
        return list

    def timestr_to_datetime(self, timestr):
        time = None
        try:
            time = datetime.strptime(timestr, '%m/%d/%Y %H:%M:%S')
        except ValueError, e:
            print('Error: %s' % (timestr))
        return time

    def load_data(self, load=False):
        if (load or not (os.path.exists(self.fname))):
            print('[INFO] Creating Data Frame from Scratch.')
            new_df = pd.read_csv(self.file_name, dtype={'Geostamp': str})
            new_df['Content'] = new_df.apply(lambda row: self.tweet_to_list(row['Content']), axis=1)
            new_df['Geostamp'] = new_df.apply(lambda row: self.geostamp_to_list(row['Geostamp']), axis=1)
            new_df['isHillary'] = new_df.apply(lambda row: bool(row['isHillary']), axis=1)
            new_df['Timestamp'] = new_df.apply(lambda row: self.timestr_to_datetime(row['Date'] + ' ' + row['Time']),
                                               axis=1)
            new_df.drop(new_df.columns[len(new_df.columns) - 1], axis=1, inplace=True)
            self.df = new_df
        else:
            print('[INFO] Using Data Frame from Pickle File.')
            self.df = pd.read_pickle(self.fname)
            # print(list(self.df.columns.values))
            # print(self.df.dtypes)
            # print(self.df.head(1))

    def analyze_data(self):
        for index, tweet in self.df.iterrows():
            # Temporary Fix
            tweet['Content'] = [term.lower() for term in tweet['Content']]

            str = ' '.join(tweet['Content'])
            unicode_tweet = unicode(str, errors='replace')
            self.corpus.append(unicode_tweet)

            filtered_list = [term for term in tweet['Content'] if not term.startswith(('#', '@'))]
            self.terms['filtered'].extend(tweet['Content'])
            self.terms['all'].extend(tweet['Content'])

            if (tweet['isHillary']):
                self.terms['hillary'].extend(tweet['Content'])
            else:
                self.terms['trump'].extend(tweet['Content'])

            if tweet['Compound'] > 0:
                self.terms['positive'].extend(tweet['Content'])
            elif tweet['Compound'] < 0:
                self.terms['negative'].extend(tweet['Content'])
            else:
                self.terms['neutral'].extend(tweet['Content'])

            if tweet['Geostamp']:
                time = tweet['Timestamp'].strftime('%m/%d/%Y %H:%M:%S').encode('utf-8').strip()
                latlang = tweet['Geostamp'][0]
                latlang[0], latlang[1] = latlang[1], latlang[0]
                coordinates = {'coordinates': latlang, 'type': 'Point'}
                geo_json_feature = {
                    'type': 'Feature',
                    'geometry': coordinates,
                    'properties': {
                        'text': unicode_tweet,
                        'created_at': time
                    }
                }
                self.geo_data['features'].append(geo_json_feature)

    def create_counters(self):
        # Print out 10 most frequent words filtered
        print('\n[INFO] Filtered Frequency:')
        self.terms_filtered_counter = Counter(self.terms['filtered'])
        for word, count in self.terms_filtered_counter.most_common(15):
            print('{0}: {1}'.format(word, count))

        # Print out 10 most unfiltered frequent words
        print('\n[INFO] Unfiltered Frequency:')
        self.terms_all_counter = Counter(self.terms['all'])
        for word, count in self.terms_all_counter.most_common(15):
            print('{0}: {1}'.format(word, count))

    def create_tfidf(self):
        # Print and generate tfidf matrix from all the tweets
        print('\n[INFO] Tf-Idf Vectors:')
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        self.tfidf_matrix = tf.fit_transform(self.corpus)
        feature_names = tf.get_feature_names()
        dense = self.tfidf_matrix.todense()
        densetweets = dense[0].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(densetweets)), densetweets) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][
                             :len(sorted_phrase_scores)]:
            print('{0: <40} {1}'.format(phrase, score))

    def find_cosine_similar(self, tfidf_matrix, index, top_n=5):
        cosine_similarities = linear_kernel(tfidf_matrix[index:index + 1], self.tfidf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
        return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    def print_cosine_similar(self):
        print('\n[INFO] Tweets Similar To: %s' % (self.corpus[20]))
        for index, score in self.find_cosine_similar(self.tfidf_matrix, 20):
            print('%.2f  ->  %s' % (score, self.corpus[index]))

    def analyze_geodata(self):
        with open('geo_data.json', 'w') as fout:
            print('\n[INFO] Dumped geo data into geo_data.json')
            fout.write(json.dumps(self.geo_data, indent=4))

def main():
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')

    username = pwd.getpwuid(os.getuid())[0]

    temp_subset_file = open('/Users/{0:s}/Dropbox/US_UK_ElectionTweets/US_all_tweets/temp_subset.csv'.format(username))
    all_tweets_file = open('/Users/{0:s}/Dropbox/US_UK_ElectionTweets/US_all_tweets/all_tweets.csv'.format(username))
    geo_tweets_file = open(
        '/Users/{0:s}/Dropbox/US_UK_ElectionTweets/geo_time_tweets_fixed/fixed_geo.csv'.format(username))
    temp_geo_tweets_file = open(
        '/Users/{0:s}/Dropbox/US_UK_ElectionTweets/geo_time_tweets_fixed/temp_geo.csv'.format(username))
    temp_geo_tweets_100000_file = open(
        '/Users/{0:s}/Dropbox/US_UK_ElectionTweets/geo_time_tweets_fixed/temp_geo_100000.csv'.format(username))
    app = App(temp_geo_tweets_file)

    parser = argparse.ArgumentParser(description='Analyze Political Twitter Data')
    parser.add_argument('-l', '--load', action='store_true', required=False,
                        help='Loads a CSV file from scratch rather than using existing Data-frame.')
    args = parser.parse_args()

    app.load_data(args.load)
    app.analyze_data()
    app.create_counters()
    app.create_tfidf()
    app.print_cosine_similar()
    # Save Dataframe
    app.df.to_pickle(app.fname)
    app.analyze_geodata()

    print()
    # pdb.set_trace()


if __name__ == '__main__':
    main()
