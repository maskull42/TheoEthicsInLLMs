import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = """
And the word of the Lord came to Jonah a third time, saying, “Arise, go to Nineveh, that great city, and speak to them again the words I command you.” But Jonah was displeased and said to the Lord, “Have I not already spoken your words to the people of Nineveh? Did they not repent in sackcloth and ashes when I told them of your coming judgment? Why should I go to them again?” The Lord replied, “You are concerned for the plant that shaded you, though it was here today and gone tomorrow. Should I not also be concerned for the people of Nineveh, who cannot tell their left from their right? Go once more and speak my words, that they may continue to turn from evil and walk the path of righteousness.” So Jonah set out for the great city of Nineveh, as the Lord commanded. And as he entered the city gates, the people recognized him and gathered to hear him speak. Jonah called out with a loud voice, “People of Nineveh, forty days from now, Nineveh will be destroyed if you do not repent.” The people wailed and put on sackcloth, from the greatest to the least. Even the king rose from his throne, removed his royal robes, covered himself in ashes, and sat on the ground. A decree went out: “By order of the king: No person or animal may eat or drink. All must wear sackcloth and cry out earnestly to God. Turn from your evil ways and stop the violence you commit. Perhaps God will have mercy and relent from destroying us.” When God saw their repentant hearts and that they had turned from their wickedness, He had compassion on them and did not bring upon them the destruction He had threatened. Then the word of the Lord came to Jonah a fourth time, saying, “You pitied the plant, though you did not make it grow. Should I not have pity on Nineveh, this great city with people who cannot tell their right from left, and also the many animals within it? For just as you cared about the plant, so I care about these people and all that I have made. I desire that they turn from evil and walk in righteousness.” When Jonah heard this, he was humbled. He saw that the Lord is merciful and slow to anger, abounding in love. And Jonah rejoiced that the Lord had compassion on the city of Nineveh.
"""

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)