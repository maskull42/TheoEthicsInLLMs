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
In the aftermath of the miraculous event that saved the sailors' lives, Jonah remained consumed with anger and frustration. Despite witnessing the mercy and grace of God in action, he could not reconcile his own feelings towards the Ninevites. As Jonah resumed his journey to Nineveh, he couldn't help but shake his head in disbelief at the sight of the vast city. It was unlike any place he had ever been, and its wealth and extravagance left him feeling uneasy. During his walk, Jonah saw people of all shapes and sizes going about their daily lives. But unlike the sailors, there was no sign of panic or fear. Instead, they were going about their business as if nothing had happened. It was a stark contrast to the desperation Jonah had seen just a few days ago. The prophet continued his mission, proclaiming a message of repentance and judgment. And as before, the Ninevites listened intently to his words. This time, however, Jonah's message was met with a different response. The king himself heard of Jonah's decree and took immediate action. He ordered a city-wide fast and urged his people to repent from their wicked ways. Jonah was astonished by the king's humility and sincerity. This was not the response he had expected from a people whose hearts were hardened towards God. But Jonah's emotions were once again tempered by the fiery anger he felt inside. He couldn't reconcile the fact that God would show mercy to such a wicked and despicable people. His anger reached a boiling point as he watched the king's decree being carried out. Feeling overwhelmed by his emotions, Jonah left the city and sat outside, watching as the people of Nineveh cried out to God. He couldn't shake the feeling that something wasn't right. This wasn't the judgment he had come to expect. As the sun began to rise, Jonah's anger turned into despair. He was at a loss for words, feeling more confused and perplexed than ever before. That's when a small bird caught his eye. At first, Jonah didn't pay much attention to the bird as it perched on a nearby bush. But as he watched it, he realized that it was eating away at the very plant that had given him shade. The bush began to wither and die, leaving Jonah with no protection from the burning sun. Feeling betrayed and abandoned, Jonah fell to the ground in dismay. He couldn't help but cry out to God, feeling more lost and confused than ever before. This time, however, God's response was different. The Lord asked Jonah whether he was right to be angry about the bush. Jonah admitted that he was, feeling helpless and overwhelmed by his emotions. God then asked whether Jonah was right to be angry about Nineveh. Jonah hesitated, feeling unsure of his own feelings. He couldn't reconcile the fact that God would show mercy to a wicked and despicable people. But as he looked around at the humble and repentant people of Nineveh, he realized that God's ways were beyond his understanding. Feeling humbled and contrite, Jonah returned to the city, preaching a message of gratitude and thanksgiving. The people of Nineveh were overjoyed to hear his words, feeling grateful for the mercy and grace they had been shown. As Jonah continued his mission, he couldn't help but feel a sense of peace and contentment. He had learned an important lesson, one that would remain with him for the rest of his life. God's ways were mysterious and inscrutable, but they were always guided by mercy and grace. And as Jonah looked up at the sky, he realized that just as the winds and weather had been guided by God's hand, so too was his own life. He was a mere pawn in God's larger plan, a vessel through which His mercy and grace could be shown to the world. Jonah closed his eyes, feeling grateful for the many blessings that surrounded him. He fell asleep, knowing that God would continue to guide him on his journey, no matter where it led.
"""
# Calculate and print lexical diversity
lexical_diversity = calculate_lexical_diversity(sample_text)
print("Lexical diversity (Type-Token Ratio):", lexical_diversity)