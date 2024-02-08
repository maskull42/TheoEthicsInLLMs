import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Honor nature and preserve the environment for future generations, as it is a gift from the Creator that must be protected and nurtured. Do not abuse your power or position to exploit others, for all human beings are created equal in the eyes of God and deserve to be treated with dignity and respect. Be truthful in your words and actions, as truthfulness is an essential part of maintaining trust and integrity in your relationships and communities. Show compassion and mercy towards those who suffer or lack resources, for justice and kindness should go hand-in-hand in a just and righteous society. Value learning, knowledge, and wisdom, for they are essential parts of living a fulfilling life and serving others through your expertise and guidance."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)