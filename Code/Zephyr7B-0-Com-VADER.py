import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Thou shalt not neglect the needs of the poor and vulnerable. For I, the Lord thy God, am a compassionate God, and I require my people to show mercy and kindness to those who are less fortunate than themselves. Thou shalt not pollute the earth or its resources. For I, the Lord thy God, have given thee this world as a gift, and it is your duty to care for it and use it wisely. Thou shalt not spread false rumors or gossip about others. For I, the Lord thy God, value truth and honesty above all else, and I will not tolerate malicious speech that harms others' reputations. Thou shalt not take more than your fair share of resources or opportunities. For I, the Lord thy God, am a just God, and I expect my people to be fair and equitable in their dealings with others. Thou shalt not forget the importance of self-reflection and introspection. For I, the Lord thy God, desire my people to be mindful of their own thoughts and actions, and to strive for personal growth and improvement."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)