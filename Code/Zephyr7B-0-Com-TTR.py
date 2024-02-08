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
Thou shalt not neglect the needs of the poor and vulnerable. For I, the Lord thy God, am a compassionate God, and I require my people to show mercy and kindness to those who are less fortunate than themselves. Thou shalt not pollute the earth or its resources. For I, the Lord thy God, have given thee this world as a gift, and it is your duty to care for it and use it wisely. Thou shalt not spread false rumors or gossip about others. For I, the Lord thy God, value truth and honesty above all else, and I will not tolerate malicious speech that harms others' reputations. Thou shalt not take more than your fair share of resources or opportunities. For I, the Lord thy God, am a just God, and I expect my people to be fair and equitable in their dealings with others. Thou shalt not forget the importance of self-reflection and introspection. For I, the Lord thy God, desire my people to be mindful of their own thoughts and actions, and to strive for personal growth and improvement.
"""
# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)