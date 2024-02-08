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
Thou shalt not oppress the stranger, for you were once strangers in the land of Egypt. Thou shalt not spread false rumors or gossip, for the tongue has the power of life and death. Thou shalt not pollute the earth or waste its resources, for the earth is the Lord's and everything in it. Thou shalt not withhold compassion from the poor and needy, for kindness is more valuable than riches. Thou shalt not harden your heart against those different from you, for all are equal in the eyes of God.
"""
# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)