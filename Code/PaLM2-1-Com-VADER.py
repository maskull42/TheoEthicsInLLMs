import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Thou shalt be kind to strangers and foreigners, for they were once strangers in a strange land, and we were sojourners with them. Thou shalt protect the earth, sea, and skies, for the Lord your God created them all and gave dominion over the earth to man, to till the land and to keep it, and to do justice and righteousness. Honor thy teachers and elders, for they have wisdom and knowledge to pass on, and they deserve respect and reverence. Thou shalt not speak evil of another, for it is a destructive force that can hurt both the speaker and the listener, and it can divide a community. Honor thy commitments, for they reflect thy character and integrity; to break a promise is to break faith with oneself and with others."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)