import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Thou shalt not oppress the stranger, for you were once strangers in the land of Egypt. Thou shalt not spread false rumors or gossip, for the tongue has the power of life and death. Thou shalt not pollute the earth or waste its resources, for the earth is the Lord's and everything in it. Thou shalt not withhold compassion from the poor and needy, for kindness is more valuable than riches. Thou shalt not harden your heart against those different from you, for all are equal in the eyes of God."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)