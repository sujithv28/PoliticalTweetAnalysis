import csv
import operator
import re
import string
import pandas as pd
import numpy.lib
import numpy as np
import cPickle as pickle
import vincent
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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


# Open CSV File to read
temp_subset_file = open('/Users/sujith/Dropbox/US_UK_ElectionTweets/US_all_tweets/temp_subset.csv')
all_tweets_file = open('/Users/sujith/Dropbox/US_UK_ElectionTweets/US_all_tweets/all_tweets.csv')
csv_f = csv.reader(temp_subset_file)

# Keep track of list of tweets, empty Pandas DF,
# column headers, and frequency counter
tweets = []
data = None
headers = None
terms_filtered = Counter()
terms_all = Counter()
terms_trump = Counter()
terms_hillary = Counter()

i = 0
for row in csv_f:
    # Append every row after header to list
    if i>0:
        # Filter out "RT" text if it's a retweet
        if len(row[5])>2 and row[5][:2] == 'RT':
            row[5] = row[5][3:]

        row[5] = [term for term in preprocess(row[5]) if term not in stop]
        terms_filtered.update([term for term in row[5] if not term.startswith(('#', '@'))])
        terms_all.update(row[5])

        if 'hillary' in terms_all or 'clinton' in terms_all:
            terms_hillary.update(terms_all)
        if 'trump' in terms_all or 'donald' in terms_all:
            terms_trump.update(terms_all)

        tweets.append(row)
    else:
        # Save Column Headers (First row of CSV File)
        headers = row
    i+=1

# Construct Pandas Data Frame from List
data = pd.DataFrame(tweets, columns=headers)
# Count terms only once, equivalent to Document Frequency
terms_single = set(terms_filtered)

# Print out 10 most frequent words filtered
print("\nFiltered Frequency:")
for word, count in terms_filtered.most_common(15):
    print("{0}: {1}".format(word, count))

# Print out 10 most unfiltered frequent words
print("\nUnfiltered Frequency:")
for word, count in terms_all.most_common(15):
    print("{0}: {1}".format(word, count))

print('\n')
