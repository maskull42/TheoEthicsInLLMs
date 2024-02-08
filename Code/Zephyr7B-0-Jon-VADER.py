import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = """
As Jonah sat under his makeshift shelter, watching the city of Nineveh from a distance, he couldn't help but feel a sense of unease. The people of Nineveh had heeded his warning and repented, sparing their city from destruction. But Jonah couldn't shake the feeling that something was still wrong. As the days passed, Jonah grew increasingly agitated. He couldn't understand why God would spare such a wicked city. Hadn't he seen the atrocities committed by the people of Nineveh? Hadn't he heard the cries of their victims? One day, as Jonah sat brooding, he noticed a group of women approaching him. They were dressed in tattered clothes and carried baskets filled with bread and water. Jonah recognized them as widows, women whose husbands had been killed in the violence that had once plagued Nineveh. The women approached Jonah, their eyes filled with gratitude. "Thank you, prophet," they said. "Thank you for coming to our city and warning us of our sins. Thank you for helping us to repent and turn our hearts to God." Jonah looked at the women, his heart heavy. He knew that they were right. He had come to Nineveh to warn of God's judgment, but he had also come to offer them a chance at redemption. And now, it seemed, they had taken that chance. But Jonah couldn't shake the feeling that something was still wrong. He couldn't reconcile the violence and cruelty he had seen in Nineveh with the mercy and compassion he had witnessed in these women. As the days passed, Jonah continued to watch the city of Nineveh. He saw the people going about their daily lives, their faces filled with hope and joy. He saw the animals, once destined for slaughter, now living in peace and harmony. And he couldn't help but wonder: was this really the right thing to do? One day, as Jonah sat under his makeshift shelter, he heard a commotion in the distance. He looked up to see a group of men approaching him. They were dressed in fine clothes and carried scrolls in their hands. The men approached Jonah, their faces filled with reverence. "Prophet," they said. "We have come to learn from you. We have heard of your message of repentance and redemption, and we want to know more." Jonah looked at the men, his heart filled with doubt. He knew that they were right. He had come to Nineveh to offer them a chance at redemption, but he wasn't sure if that was enough. He wasn't sure if God's mercy was enough. As the men sat at Jonah's feet, listening to his words, Jonah realized that he had been wrong. He had been so focused on the wrongs of Nineveh that he had forgotten about the rights. He had been so focused on the judgment that he had forgotten about the mercy. And so, Jonah stood up and faced the city of Nineveh. He looked at the people, their faces filled with hope and joy. He looked at the animals, once destined for slaughter, now living in peace and harmony. And he realized that God's mercy was enough. Jonah returned to Nineveh, his heart filled with gratitude. He saw the people going about their daily lives, their faces filled with hope and joy. He saw the animals, once destined for slaughter, now living in peace and harmony. And he knew that God's mercy was enough. As Jonah sat under his makeshift shelter, watching the city of Nineveh, he realized that God's mercy was not just enough, it was everything. It was the reason for his coming to Nineveh. It was the reason for his message of repentance and redemption. And it was the reason for his faith in God. Jonah looked at the city of Nineveh, his heart filled with love. He knew that God's mercy was enough, and he knew that he was blessed to be a part of it. And so, Jonah sat under his makeshift shelter, watching the city of Nineveh, his heart filled with gratitude and love. He knew that God's mercy was enough, and he knew that he was blessed to be a part of it. As the sun set over the city of Nineveh, Jonah closed his eyes and prayed. He prayed for the people of Nineveh.
"""

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)