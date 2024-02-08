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
Thou shalt not neglect the stranger nor the sojourner among you, for you were strangers in the land of Egypt. As you show kindness to those who dwell within your gates, so shall kindness be shown unto you. Thou shalt not turn thy back on the poor and the needy; open thy hand wide unto thy brother, to the afflicted and destitute in thy land. In giving, thou shalt reflect the bounty of the Lord thy God. Thou shalt not speak with deceit nor harbor guile in thy heart. Let thy words be as pure as the waters of Eden, and let truth be the mantle that adorns thy soul. Thou shalt not indulge in wrath or hold onto anger against thy kin. Seek reconciliation and pursue peace, for in forgiveness lies the path to righteousness. Thou shalt not despoil the earth, the seas, nor the skies, for the Lord has entrusted them unto your care. Be stewards of creation, that it may sustain your children and your children's children unto the thousandth generation.
"""

# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)