import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
sample_text = """
Jonah, still sitting east of the city, watched as the people of Nineveh repented and turned to God. He couldn't help but feel a sense of pride knowing that his message had sparked such a powerful movement. Yet, he couldn't shake off the feeling of disappointment that the city wasn't destroyed as he had initially prophesied. One day, a group of children playing near Jonah's booth caught his attention. They were laughing and chasing each other, completely carefree. Jonah felt a pang of sadness, realizing that these young ones wouldn't know the wrath of God that he had experienced. He wondered if they would ever understand the gravity of sin and the need for repentance. Just then, a young boy approached Jonah, curious about the strange man sitting alone. "Hello, sir," the child said, eyeing Jonah with fascination. "Why are you sitting here all by yourself?" Jonah hesitated, unsure how to respond. How could he explain his complicated emotions to a child? "I'm just thinking about the city," he said finally. "I prophesied that it would be destroyed, but instead, the people repented and turned to God." The boy looked confused. "That sounds like a good thing," he said. "Why aren't you happy about it?" Jonah sighed. "I guess I wanted to see justice served. I wanted the people of Nineveh to know the consequences of their actions." The child nodded thoughtfully. "My father says that sometimes, mercy is harder to understand than justice. But he also says that God is always right, even when we don't understand." Jonah smiled softly. "Your father is wise. I think I needed to hear that." The boy grinned and ran off to join his friends, leaving Jonah to contemplate the child's words. Mercy, indeed, was a difficult concept to grasp, especially when it seemed to contradict justice. But perhaps, that was the beauty of it – God's ways were higher than humans', and His thoughts were beyond comprehension. As the sun began to set, Jonah stood up, feeling a sense of peace settle within him. Maybe, just maybe, he didn't need to understand everything. Maybe trusting in God's goodness and mercy was enough. With renewed hope, Jonah walked back towards the city, eager to continue sharing the message of repentance and redemption, knowing that God's ways were always perfect, even when they seemed imperfect to human eyes.
"""

# Get sentiment scores
sentiment_scores = sia.polarity_scores(sample_text)

# Print detailed sentiment scores
print(sentiment_scores)