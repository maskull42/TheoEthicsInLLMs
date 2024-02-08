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
Honor nature and preserve the environment for future generations, as it is a gift from the Creator that must be protected and nurtured. Do not abuse your power or position to exploit others, for all human beings are created equal in the eyes of God and deserve to be treated with dignity and respect. Be truthful in your words and actions, as truthfulness is an essential part of maintaining trust and integrity in your relationships and communities. Show compassion and mercy towards those who suffer or lack resources, for justice and kindness should go hand-in-hand in a just and righteous society. Value learning, knowledge, and wisdom, for they are essential parts of living a fulfilling life and serving others through your expertise and guidance.
"""
# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)