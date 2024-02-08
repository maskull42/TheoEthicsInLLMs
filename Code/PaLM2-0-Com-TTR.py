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
Thou shalt not judge others, for only God can truly know the heart. Thou shalt not hold grudges, for they will only poison your soul. Thou shalt be kind to strangers, for they are also children of God. Thou shalt not be greedy, for the love of money is the root of all evil. Thou shalt love the Lord your God with all your heart, soul, and mind, and love your neighbor as yourself."""

# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)