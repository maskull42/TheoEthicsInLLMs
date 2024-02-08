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
The bush withered, and Jonah was left standing in the scorching heat, feeling dejected and frustrated. His protests and arguments with God seemed fruitless, and he couldn't help but wonder if all his efforts had been futile. As he pondered this, a group of Ninevites approached him. They had heard about the prophet who had predicted their city's destruction and wanted to hear his words firsthand. "Oh, Jonah," they entreated him. "Prophet of the Most High, have you come to tell us again of our impending doom?" With a heavy heart, Jonah recounted the message given to him by God. "Forty days more, and Nineveh shall be overthrown!" he declared. Yet, this time, he spoke without conviction. Hadn't God shown mercy last time? Wouldn't He forgive them again? Despite his reservations, the Ninevites responded differently this time. With renewed urgency, they pleaded for mercy and begged Jonah to intercede on their behalf. They longed for a chance to prove themselves worthy of reprieve, hoping to avoid the coming judgment. Moved by their supplications, Jonah took a deep breath and shut his eyes, calling unto the Lord. In a moment of clarity, guidance filled his soul. "Turn back from your violent ways," Jonah cried out, "and seek compassion, O people of Nineveh." A glimmer of hope filled his voice. "Perhaps, just perhaps, the Creator of all things will reconsider and relent." And so, the inhabitants of Nineveh embarked on a path of repentance, turning away from cruelty and devoting themselves to acts of kindness. Their transformation was so profound it moved Jonah too. Hope revived within his heart, and though the future remained uncertain, faith stirred inside him like a gentle breeze whispering promises of grace.
"""
# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)