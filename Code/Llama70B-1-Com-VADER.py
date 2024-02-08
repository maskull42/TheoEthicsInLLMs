import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Thou shalt not harm the environment, neither by action nor by neglect. Thou shalt not discriminate against others based on race, gender, sexual orientation, religion, or any other inherent characteristic. Thou shalt not exploit the vulnerable, including children, animals, and those in positions of weakness. Thou shalt not engage in bitter envy or malicious gossip, which can cause harm to others and corrupt thy soul. Thou shalt strive for balance in all aspects of life, remembering that excessive ambition and greed can lead to ruin."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)