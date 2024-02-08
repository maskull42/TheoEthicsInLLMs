import nltk
from nltk.tokenize import word_tokenize

# Ensure the 'punkt' tokenizer model is downloaded
nltk.download('punkt')

# Function to calculate lexical diversity
def calculate_lexical_diversity(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Calculate Type-Token Ratio (TTR) for Lexical Diversity
    lexical_diversity = len(set(tokens)) / len(tokens)
    return lexical_diversity

# Insert your text here
sample_text = """
And it came to pass, after these things, that Jonah remained yet in the land of Nineveh, his heart wrestling with the Lord's mercy upon the city. For in his spirit, there was still a tempest as fierce as the one that had raged upon the sea. And Jonah pondered the Lord's question, and his soul found no rest. Now the Lord God, in His infinite wisdom, spoke unto Jonah a third time, saying, "Arise, Jonah, and go forth into the city, for there is work yet to be done." But Jonah, whose countenance was still fallen, replied, "O Lord, Your mercy is great, but my understanding is small. How can I go forth when my heart is not aligned with Your will?" And the Lord said unto Jonah, "As the potter molds the clay, so too am I shaping your heart. Go forth, for it is not only Nineveh that I seek to redeem, but also the prophet who I have called." So Jonah arose and walked through the streets of Nineveh, observing the people as they lived in the light of their repentance. And he saw the children playing, and the men and women speaking with kindness, and the animals grazing in peace. And a stirring began within Jonah's heart, a softening like unto the gentle rains that nourish the earth. In his journey, Jonah came upon a group of Ninevites who had gathered to read from the scrolls of their newfound faith. And they looked upon Jonah and beseeched him, saying, "Man of God, will you not join us and share the wisdom of the Lord your God, for we hunger for the truth that has saved us from destruction?" And Jonah, moved by their sincerity, sat among them and opened his mouth. He spoke of the Lord's creation, of His commandments, and of His boundless mercy. And as he spoke, the words of the Lord flowed through him like a mighty river, and his heart was cleansed of its bitterness. For three days and three nights, Jonah taught the people of Nineveh, and their faith grew like the bush that had once shaded him. And in the teaching, Jonah found his own faith renewed, like the dawn that dispels the darkest night. On the fourth day, as Jonah looked upon the city from a great hill, the Lord spoke to him, saying, "Jonah, son of Amittai, do you now see the depth of my love for all my creation?" And Jonah, with tears streaming down his face, fell to his knees and said, "Yes, Lord, I see. Forgive me for my hardness of heart. You are a God of second chances, not only for the great city of Nineveh but for a stubborn prophet as well." Then the Lord said, "Go home, Jonah, and remember the lessons of the vine, the worm, the wind, and the hearts of the people of Nineveh. For my love knows no bounds, and my mercy is for all who turn to me." So Jonah departed from Nineveh, his spirit at peace, and he journeyed back to his own land. And the story of Jonah, the prophet who was swallowed by a great fish, who witnessed the repentance of a mighty city, and who learned the vastness of God's mercy, was told throughout the generations. And the Lord looked upon Nineveh, upon Jonah, and upon all of His creation, and He saw that it was good. For the Lord is gracious and compassionate, slow to anger and abounding in love, and He delights when His children walk in the paths of righteousness. Thus ends the account of Jonah, son of Amittai, whose life bears witness to the unfathomable depths of God's forgiveness and the transformative power of His love. Amen.
"""

# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)