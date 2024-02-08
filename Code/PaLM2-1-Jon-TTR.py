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
Now, the word of the Lord came to Jonah a third time, saying, 'Arise, go to the great city of Nineveh and speak to it the message that I will tell you.'" Jonah, weary from his previous experiences, yet obedient to the divine call, sets out to Nineveh, a vast metropolis known for its wickedness. As Jonah journeys deeper into the city, the weight of his mission weighs heavily on him. He struggles with his emotions, wavering between hope for the people's repentance and dread of the potential consequences. Days turn into weeks, and Jonah's anguish intensifies as he observes the inhabitants of Nineveh carrying on with their lives seemingly unfazed by his warnings. However, a stirring takes place in the heart of the city. The message of impending doom and the prophet's unwavering conviction begin to resonate within the people. Remarkably, a collective awakening occurs as the people of Nineveh, from the highest officials to the lowliest citizens, humble themselves before God. They declare a fast and clothe themselves in sackcloth, fervently seeking divine mercy. News of the city's repentance reaches the ears of the king, who, moved by the genuine remorse of his people, joins them in their acts of repentance. He issues a decree, urging every living creature within the city to join in the fervent plea for forgiveness. Word spreads throughout the land, reaching the ears of neighboring nations. Astonished by the spiritual transformation unfolding in Nineveh, they too are compelled to seek the Lord's mercy. The news of Nineveh's repentance creates ripples of hope and inspires neighboring nations to reflect on their own spiritual standing. Jonah, witnessing the profound change within the city that he once feared to enter, can no longer deny the transformative power of divine compassion and forgiveness. His anger towards the people's seeming indifference fades away, replaced by awe and joy.
"""

# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)