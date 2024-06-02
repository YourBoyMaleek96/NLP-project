from twikit import Client
import pandas as pd
from transformers import pipeline

# Initialize BERT 
sentiment_analysis_pipeline = pipeline("sentiment-analysis")

client = Client('en-US')

# Twitter info
client.login(
    auth_info_1='MalikFreem4293',
    password='K!ngjames1996',
)

client.save_cookies('cookies.json')
client.load_cookies(path='cookies.json')

#get tweets from Lebron
user = client.get_user_by_screen_name('KingJames')
tweets = user.get_tweets('Tweets', count=20)  

positive_tweets = []
negative_tweets = []

for tweet in tweets:
    tweet_text = tweet.full_text
    # Perform sentiment analysis on the tweet
    sentiment = sentiment_analysis_pipeline(tweet_text)[0]
    if sentiment['label'] == 'POSITIVE' and len(positive_tweets) < 5:
        positive_tweets.append((tweet_text, tweet.created_at))
    elif sentiment['label'] == 'NEGATIVE' and len(negative_tweets) < 5:
        negative_tweets.append((tweet_text, tweet.created_at))

print("Positive Tweets:")
for tweet, date_time in positive_tweets:
    print(f"{date_time}: {tweet}")

print("\nNegative Tweets:")
for tweet, date_time in negative_tweets:
    print(f"{date_time}: {tweet}")
