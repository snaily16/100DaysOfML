import tweepy
from textblob import TextBlob

consumer_key = 'CONSUMER KEY'
consumer_secret ='CONSUMER SECRET KEY'

access_token = 'ACCESS TOKEN'
access_token_secret = 'ACCESS TOKEN SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

def get_label():
	return input('Enter label: ')

def get_sentiment(tweets):
	analysis = TextBlob(tweets)
	if analysis.sentiment.polarity > 0:
		return 'positive'
	elif analysis.sentiment.polarity < 0:
		return 'negative'
	else:
		return 'neutral'

data=[]
name = get_label()
public_tweets = api.search(name, count=100)
parsed_tweet ={'text':'NaN','user':'NaN','sentiment':'NaN'}

with open('%s_tweets.csv' % name, 'w') as tweets_data:
	writer = csv.DictWriter(tweets_data, parsed_tweet.keys())
	writer.writeheader()
	for tweet in public_tweets:
		parsed_tweet['text']=tweet.text
		parsed_tweet['user']=tweet.user.screen_name
		parsed_tweet['sentiment'] = get_sentiment(tweet.text)
		writer.writerow(parsed_tweet)

