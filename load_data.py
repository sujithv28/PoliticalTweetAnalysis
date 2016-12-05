import csv

file = open('/Users/sujith/Dropbox/US_UK_ElectionTweets/US_all_tweets/temp_subset.csv')
csv_f = csv.reader(file)

tweets = []
for row in csv_f:
  tweets.append(row)
