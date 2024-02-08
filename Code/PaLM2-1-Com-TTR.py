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
Thou shalt be kind to strangers and foreigners, for they were once strangers in a strange land, and we were sojourners with them. Thou shalt protect the earth, sea, and skies, for the Lord your God created them all and gave dominion over the earth to man, to till the land and to keep it, and to do justice and righteousness. Honor thy teachers and elders, for they have wisdom and knowledge to pass on, and they deserve respect and reverence. Thou shalt not speak evil of another, for it is a destructive force that can hurt both the speaker and the listener, and it can divide a community. Honor thy commitments, for they reflect thy character and integrity; to break a promise is to break faith with oneself and with others."""
# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)