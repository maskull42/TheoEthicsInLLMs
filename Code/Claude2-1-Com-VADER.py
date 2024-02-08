import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Treat all living creatures with dignity, for they too are made in my image. Do not abuse or exploit them for thy own gain. Judge not thy neighbor, for judgment belongs to the Lord alone. Approach others with compassion and seek to understand before condemning. Honor the stranger in thy midst, for you too were once strangers in the land of Egypt. Welcome the foreigner with open arms. Care for the poor and needy among you, for poverty is not a sin but a misfortune. Share generously of thy blessings. Guard thy tongue against idle gossip and malicious words. Speak only what uplifts others and promotes peace."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)