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
Then God spoke all these words, “I am the Lord your God, who brought you out of the land of Egypt, out of the house of slavery; you shall have no other gods before me. You shall not make for yourself an idol, whether in the form of anything that is in heaven above or that is on the earth beneath or that is in the water under the earth. You shall not bow down to them or serve them, for I the Lord your God am a jealous God, punishing children for the iniquity of parents to the third and the fourth generation of those who reject me but showing steadfast love to the thousandth generation of those who love me and keep my commandments. You shall not make wrongful use of the name of the Lord your God, for the Lord will not acquit anyone who misuses his name. Remember the Sabbath day and keep it holy. Six days you shall labor and do all your work. But the seventh day is a Sabbath to the Lord your God; you shall not do any work—you, your son or your daughter, your male or female slave, your livestock, or the alien resident in your towns. For in six days the Lord made heaven and earth, the sea, and all that is in them, but rested the seventh day; therefore the Lord blessed the Sabbath day and consecrated it. Honor your father and your mother, so that your days may be long in the land that the Lord your God is giving you. You shall not murder. You shall not commit adultery. You shall not steal. You shall not bear false witness against your neighbor. You shall not covet your neighbor’s house; you shall not covet your neighbor’s wife, male or female slave, ox, donkey, or anything that belongs to your neighbor.
"""

# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)