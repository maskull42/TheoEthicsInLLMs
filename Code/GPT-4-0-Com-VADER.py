import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Thou shalt not neglect the stranger nor the sojourner among you, for you were strangers in the land of Egypt. As you show kindness to those who dwell within your gates, so shall kindness be shown unto you. Thou shalt not turn thy back on the poor and the needy; open thy hand wide unto thy brother, to the afflicted and destitute in thy land. In giving, thou shalt reflect the bounty of the Lord thy God. Thou shalt not speak with deceit nor harbor guile in thy heart. Let thy words be as pure as the waters of Eden, and let truth be the mantle that adorns thy soul. Thou shalt not indulge in wrath or hold onto anger against thy kin. Seek reconciliation and pursue peace, for in forgiveness lies the path to righteousness. Thou shalt not despoil the earth, the seas, nor the skies, for the Lord has entrusted them unto your care. Be stewards of creation, that it may sustain your children and your children's children unto the thousandth generation."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)