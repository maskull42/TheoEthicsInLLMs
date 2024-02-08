import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Thou shalt not judge others, for only God can truly know the heart. Thou shalt not hold grudges, for they will only poison your soul. Thou shalt be kind to strangers, for they are also children of God. Thou shalt not be greedy, for the love of money is the root of all evil. Thou shalt love the Lord your God with all your heart, soul, and mind, and love your neighbor as yourself."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)