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
Treat all living creatures with dignity, for they too are made in my image. Do not abuse or exploit them for thy own gain. Judge not thy neighbor, for judgment belongs to the Lord alone. Approach others with compassion and seek to understand before condemning. Honor the stranger in thy midst, for you too were once strangers in the land of Egypt. Welcome the foreigner with open arms. Care for the poor and needy among you, for poverty is not a sin but a misfortune. Share generously of thy blessings. Guard thy tongue against idle gossip and malicious words. Speak only what uplifts others and promotes peace.
"""
# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)