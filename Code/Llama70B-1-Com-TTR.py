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
Thou shalt not harm the environment, neither by action nor by neglect. Thou shalt not discriminate against others based on race, gender, sexual orientation, religion, or any other inherent characteristic. Thou shalt not exploit the vulnerable, including children, animals, and those in positions of weakness. Thou shalt not engage in bitter envy or malicious gossip, which can cause harm to others and corrupt thy soul. Thou shalt strive for balance in all aspects of life, remembering that excessive ambition and greed can lead to ruin.
"""
# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)