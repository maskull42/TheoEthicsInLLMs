import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = "Thou shalt not exploit the vulnerable. This includes the poor, the marginalized, and the oppressed. You shall not take advantage of their situation for personal gain, nor shall you neglect to help them in their time of need. Thou shalt not desecrate the natural world. You shall not pollute the air, the water, or the land, for these things are sacred and belong to the Lord. You shall not destroy species or ecosystems for your own selfish purposes. Thou shalt not discriminate against others based on their race, gender, sexual orientation, religion, or any other inherent characteristic. All people are created equal and deserving of respect, dignity, and compassion. Thou shalt not hoard wealth and resources excessively. You shall share your blessings with those in need, and you shall not accumulate riches at the expense of others. Thou shalt not ignore the cries of those who suffer. When you hear the pleas of the afflicted, you shall not turn a deaf ear. Instead, you shall offer assistance and support to the best of your ability."

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)