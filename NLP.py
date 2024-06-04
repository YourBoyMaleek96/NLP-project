from twikit import Client
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt

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

# Define usernames and points per game for each player
players = {
    'LeBron James': {'username': 'KingJames', 'points_per_game': 25.7},
    'Stephen Curry': {'username': 'StephenCurry30', 'points_per_game': 32},
    'Kevin Durant': {'username': 'KDTrey5', 'points_per_game': 22}
}

# Define the desired count of tweets for each user
tweet_count = 20  # You can adjust this number as needed

# Fetch tweets and perform sentiment analysis for each player
all_sentiments = {}
for player, info in players.items():
    user = client.get_user_by_screen_name(info['username'])
    tweets = user.get_tweets('Tweets', count=tweet_count)
    if tweets:
        print(f"{len(tweets)} tweets fetched successfully for {player}.")
        sentiments = []
        for tweet in tweets:
            tweet_text = tweet.full_text
            print(f"Tweet: {tweet_text}")  # Debugging print
            sentiment = sentiment_analysis_pipeline(tweet_text)[0]['label']
            print(f"Sentiment: {sentiment}")  # Debugging print
            sentiments.append(sentiment)
        all_sentiments[player] = sentiments
        print(f"Sentiment distribution for {player}: {np.unique(sentiments, return_counts=True)}")  # Debugging print
    else:
        print(f"No tweets fetched for {player}. Check your connection or credentials.")

# Calculate sentiment distribution for each player
sentiment_distribution = {}
for player, sentiments in all_sentiments.items():
    unique_sentiments, sentiment_counts = np.unique(sentiments, return_counts=True)
    sentiment_distribution[player] = dict(zip(unique_sentiments, sentiment_counts))

# Ensure all players have sentiment data for all three categories
for player in players.keys():
    for sentiment_category in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        if sentiment_category not in sentiment_distribution[player]:
            sentiment_distribution[player][sentiment_category] = 0

# Create scatter plot for PPG vs. number of positive tweets
plt.figure(figsize=(10, 6))
for player, info in players.items():
    ppg = info['points_per_game']
    positive_tweets = sentiment_distribution[player]['POSITIVE']
    plt.scatter(ppg, positive_tweets, label=player)

plt.xlabel('Points Per Game (PPG)')
plt.ylabel('Number of Positive Tweets')
plt.title('Correlation between PPG and Number of Positive Tweets')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create scatter plot for PPG vs. number of negative tweets
plt.figure(figsize=(10, 6))
for player, info in players.items():
    ppg = info['points_per_game']
    negative_tweets = sentiment_distribution[player]['NEGATIVE']
    plt.scatter(ppg, negative_tweets, label=player)

plt.xlabel('Points Per Game (PPG)')
plt.ylabel('Number of Negative Tweets')
plt.title('Correlation between PPG and Number of Negative Tweets')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
