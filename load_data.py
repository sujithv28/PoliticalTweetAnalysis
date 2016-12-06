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
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Open CSV File to read
temp_subset_file = open('/Users/sujith/Dropbox/US_UK_ElectionTweets/US_all_tweets/temp_subset.csv')
all_tweets_file = open('/Users/sujith/Dropbox/US_UK_ElectionTweets/US_all_tweets/all_tweets.csv')
csv_f = csv.reader(temp_subset_file)
fname = 'tweets_dataframe'

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

# Keep track of list of tweets, empty Pandas DF,
# column headers, and frequency counter
twitter_data = []
corpus = []
data = None
headers = None
terms_filtered = []
terms_all = []
terms_trump = []
terms_hillary = []

i = 0
for row in csv_f:
    # Append every row after header to list
    if i>0:
        if (i%1000 == 0):
            print('Analyzed %d/10000' % (i))
        # Filter out "RT" text if it's a retweet
        if len(row[5])>2 and row[5][:2] == 'RT':
            row[5] = row[5][3:]

        unicode_tweet = unicode(row[5], errors='replace')
        corpus.append(unicode_tweet)

        row[5] = [term for term in preprocess(row[5]) if term not in stop]
        terms_filtered.extend([term for term in row[5] if not term.startswith(('#', '@'))])
        terms_all.extend(row[5])

        if 'hillary' in terms_all or 'clinton' in terms_all:
            terms_hillary.extend(terms_all)
        if 'trump' in terms_all or 'donald' in terms_all:
            terms_trump.extend(terms_all)

        twitter_data.append(row)
    else:
        # Save Column Headers (First row of CSV File)
        headers = row
    i+=1

# Construct Pandas Data Frame from List
data = pd.DataFrame(twitter_data, columns=headers)
# Count terms only once, equivalent to Document Frequency
terms_single = set(terms_filtered)

# Print out 10 most frequent words filtered
print('\nFiltered Frequency:')
terms_filtered_counter = Counter(terms_filtered)
for word, count in terms_filtered_counter.most_common(15):
    print('{0}: {1}'.format(word, count))

# Print out 10 most unfiltered frequent words
print('\nUnfiltered Frequency:')
terms_all_counter = Counter(terms_filtered)
for word, count in terms_all_counter.most_common(15):
    print('{0}: {1}'.format(word, count))

# Print and generate tfidf matrix from all the tweets
print('\nTf-Idf Vectors:')
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
tfidf_matrix =  tf.fit_transform(corpus)
feature_names = tf.get_feature_names()
dense = tfidf_matrix.todense()
densetweets = dense[0].tolist()[0]
phrase_scores = [pair for pair in zip(range(0, len(densetweets)), densetweets) if pair[1] > 0]
sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
   print('{0: <20} {1}'.format(phrase, score))

print('\n')

# SAVE
# np.save(open(fname, 'w'), data)
# if len(data.shape) == 2:
#     meta = data.index,data.columns
# elif len(data.shape) == 1:
#     meta = (data.index,)
# else:
#     raise ValueError('save_pandas: Cannot save this type')
# s = pickle.dumps(meta)
# s = s.encode('string_escape')
# with open(fname, 'a') as f:
#     f.seek(0, 2)
#     f.write(s)

# LOAD
# values = np.load(fname, mmap_mode=mmap_mode)
# with open(fname) as f:
#     numpy.lib.format.read_magic(f)
#     numpy.lib.format.read_array_header_1_0(f)
#     f.seek(values.dtype.alignment*values.size, 1)
#     meta = pickle.loads(f.readline().decode('string_escape'))
# if len(meta) == 2:
#     return pd.DataFrame(values, index=meta[0], columns=meta[1])
# elif len(meta) == 1:
#     return pd.Series(values, index=meta[0])
