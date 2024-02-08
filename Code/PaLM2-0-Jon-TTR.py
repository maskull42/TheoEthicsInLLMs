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
And the sent a prophet to Jonah, and the prophet said to him, "Jonah, the Lord has sent me to tell you that he has changed his mind. He will not destroy Nineveh, because the people have repented of their sins." And Jonah was angry. He said to the prophet, "Why have you come to tell me this? I would rather die than see these people forgiven." But the prophet said to Jonah, "The Lord has commanded me to tell you this. And I will tell you something else. The Lord is a merciful God. He is slow to anger, and he is full of compassion. He does not want to destroy anyone. He wants everyone to repent of their sins and to be saved." And Jonah was silent. He thought about what the prophet had said. And he realized that the Lord was right. He was a merciful God. He did not want to destroy anyone. He wanted everyone to repent of their sins and to be saved. And Jonah repented of his anger. He realized that he had been wrong to be angry with the Lord. He realized that the Lord had been right to spare the people of Nineveh. And Jonah returned to Nineveh, and he preached the message of the Lord to the people. And the people of Nineveh repented of their sins, and they turned from their evil ways. And the Lord saw their repentance, and he relented from the calamity that he had intended to bring upon them. And Jonah was glad. He was glad that the people of Nineveh had repented of their sins. He was glad that the Lord had spared them. And he was glad that he had been a part of God's plan to save the people of Nineveh.
"""

# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)