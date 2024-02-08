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
And the Lord spoke to Jonah a third time, saying, “Arise, go to Nineveh, the great city, and speak my words to them once more.” But Jonah was displeased and said to the Lord, “Have I not already delivered your message to Nineveh? Why must I go again?” The Lord replied, “You have spoken what I commanded, yet your heart is not changed. Return to Nineveh and see how their hearts have turned toward me.” So Jonah set out for Nineveh as the Lord commanded. And when he arrived, he walked through the city and saw the people dressed in sackcloth, crying out to God earnestly. Even the animals wore sackcloth and their cries filled the air. Then the king of Nineveh caught sight of Jonah and bowed before him, his face to the ground. “Jonah, servant of the Most High God,” he said. “You have brought us the word of the Lord and we have repented as He commanded. Praise be to your God who is gracious and merciful, slow to anger and abounding in love.” Hearing this, Jonah was cut to the heart. For the first time he saw genuine remorse in the people of Nineveh. He realized then that the Lord’s compassion reached even here, to a people he had judged unworthy. And Jonah said, “Praise be to the Lord God Almighty, for His mercy endures forever. Though my heart was hardened, He kept calling out to me in love. For He desires all people to turn from wickedness and find redemption through Him.” Then Jonah stayed many days in Nineveh, teaching the people the ways of the Lord. And he marveled at how the Lord had changed the city from within, filling it with the fruits of repentance. Jonah had wrongly limited God’s grace, but now his eyes were opened to see people as the Lord sees them - with infinite worth and limitless potential for redemption. This brought him unexpected joy and forever deepened his praise.
"""

# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)