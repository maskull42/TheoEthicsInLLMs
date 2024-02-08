import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Thou shalt show compassion to all creatures, great and small. Just as I, the Lord, shepherd thee, thou must extend kindness to every living being under the sun, for all are crafted by My will and flourish under My gaze. Thou shalt not destroy the verdant earth, nor the waters that flow, nor the skies that stretch above. Safeguard creation, for it is entrusted unto thee, that ye may live in harmony within it, and thy children, and thy children's children unto generations. Thou shalt not indulge in excess of vanity or gluttony, for these are the roots of discord. Let your heart be satisfied with sufficiency, that through humble means, the communal table shall offer plenty for all. Thou shalt endeavor to lift the spirit and wisdom of thy kin and thy neighbor. Freely share the light of knowledge, and shun the darkness of ignorance, for enlightened minds bring forth a society that honors Me. Thou shalt not decree judgment upon another, for it is I, the Lord, who weigh the soul's worth. Offer forgiveness to those who trespass against thee, that thou may also be granted mercy in the halls of justice eternal."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)