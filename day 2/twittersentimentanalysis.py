import tweepy
from textblob import TextBlob

consumer_key = 'CONSUMER KEY'
consumer_secret ='CONSUMER SECRET KEY'

access_token = 'ACCESS TOKEN'
access_token_secret = 'ACCESS TOKEN SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('100DaysOfMLCode')

for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)
	print("")
