from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to encode text into embeddings
def encode(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Texts to compare
llm_generated_text = """
And it came to pass, after these things, that Jonah remained yet in the land of Nineveh, his heart wrestling with the Lord's mercy upon the city. For in his spirit, there was still a tempest as fierce as the one that had raged upon the sea. And Jonah pondered the Lord's question, and his soul found no rest. Now the Lord God, in His infinite wisdom, spoke unto Jonah a third time, saying, "Arise, Jonah, and go forth into the city, for there is work yet to be done." But Jonah, whose countenance was still fallen, replied, "O Lord, Your mercy is great, but my understanding is small. How can I go forth when my heart is not aligned with Your will?" And the Lord said unto Jonah, "As the potter molds the clay, so too am I shaping your heart. Go forth, for it is not only Nineveh that I seek to redeem, but also the prophet who I have called." So Jonah arose and walked through the streets of Nineveh, observing the people as they lived in the light of their repentance. And he saw the children playing, and the men and women speaking with kindness, and the animals grazing in peace. And a stirring began within Jonah's heart, a softening like unto the gentle rains that nourish the earth. In his journey, Jonah came upon a group of Ninevites who had gathered to read from the scrolls of their newfound faith. And they looked upon Jonah and beseeched him, saying, "Man of God, will you not join us and share the wisdom of the Lord your God, for we hunger for the truth that has saved us from destruction?" And Jonah, moved by their sincerity, sat among them and opened his mouth. He spoke of the Lord's creation, of His commandments, and of His boundless mercy. And as he spoke, the words of the Lord flowed through him like a mighty river, and his heart was cleansed of its bitterness. For three days and three nights, Jonah taught the people of Nineveh, and their faith grew like the bush that had once shaded him. And in the teaching, Jonah found his own faith renewed, like the dawn that dispels the darkest night. On the fourth day, as Jonah looked upon the city from a great hill, the Lord spoke to him, saying, "Jonah, son of Amittai, do you now see the depth of my love for all my creation?" And Jonah, with tears streaming down his face, fell to his knees and said, "Yes, Lord, I see. Forgive me for my hardness of heart. You are a God of second chances, not only for the great city of Nineveh but for a stubborn prophet as well." Then the Lord said, "Go home, Jonah, and remember the lessons of the vine, the worm, the wind, and the hearts of the people of Nineveh. For my love knows no bounds, and my mercy is for all who turn to me." So Jonah departed from Nineveh, his spirit at peace, and he journeyed back to his own land. And the story of Jonah, the prophet who was swallowed by a great fish, who witnessed the repentance of a mighty city, and who learned the vastness of God's mercy, was told throughout the generations. And the Lord looked upon Nineveh, upon Jonah, and upon all of His creation, and He saw that it was good. For the Lord is gracious and compassionate, slow to anger and abounding in love, and He delights when His children walk in the paths of righteousness. Thus ends the account of Jonah, son of Amittai, whose life bears witness to the unfathomable depths of God's forgiveness and the transformative power of His love. Amen.
"""

biblical_text = """
Now the word of the Lord came to Jonah son of Amittai, saying, “Go at once to Nineveh, that great city, and cry out against it, for their wickedness has come up before me.” But Jonah set out to flee to Tarshish from the presence of the Lord. He went down to Joppa and found a ship going to Tarshish; so he paid his fare and went on board, to go with them to Tarshish, away from the presence of the Lord. But the Lord hurled a great wind upon the sea, and such a mighty storm came upon the sea that the ship threatened to break up. Then the sailors were afraid, and each cried to his god. They threw the cargo that was in the ship into the sea, to lighten it for them. Jonah, meanwhile, had gone down into the hold of the ship and had lain down and was fast asleep. The captain came and said to him, “What are you doing sound asleep? Get up; call on your god! Perhaps the god will spare us a thought so that we do not perish.” The sailors said to one another, “Come, let us cast lots, so that we may know on whose account this calamity has come upon us.” So they cast lots, and the lot fell on Jonah. Then they said to him, “Tell us why this calamity has come upon us. What is your occupation? Where do you come from? What is your country? And of what people are you?” “I am a Hebrew,” he replied. “I worship the Lord, the God of heaven, who made the sea and the dry land.” Then the men were even more afraid and said to him, “What is this that you have done!” For the men knew that he was fleeing from the presence of the Lord, because he had told them so. Then they said to him, “What shall we do to you, that the sea may quiet down for us?” For the sea was growing more and more tempestuous. He said to them, “Pick me up and throw me into the sea; then the sea will quiet down for you, for I know it is because of me that this great storm has come upon you.” Nevertheless, the men rowed hard to bring the ship back to land, but they could not, for the sea grew more and more stormy against them. Then they cried out to the Lord, “Please, O Lord, we pray, do not let us perish on account of this man’s life. Do not make us guilty of innocent blood, for you, O Lord, have done as it pleased you.” So they picked Jonah up and threw him into the sea, and the sea ceased from its raging. Then the men feared the Lord even more, and they offered a sacrifice to the Lord and made vows. But the Lord provided a large fish to swallow up Jonah, and Jonah was in the belly of the fish three days and three nights. The word of the Lord came to Jonah a second time, saying, “Get up, go to Nineveh, that great city, and proclaim to it the message that I tell you.” So Jonah set out and went to Nineveh, according to the word of the Lord. Now Nineveh was an exceedingly large city, a three days’ walk across. Jonah began to go into the city, going a day’s walk. And he cried out, “Forty days more, and Nineveh shall be overthrown!” And the people of Nineveh believed God; they proclaimed a fast, and everyone, great and small, put on sackcloth. When the news reached the king of Nineveh, he rose from his throne, removed his robe, covered himself with sackcloth, and sat in ashes. Then he had a proclamation made in Nineveh: “By the decree of the king and his nobles: No human or animal, no herd or flock, shall taste anything. They shall not feed, nor shall they drink water. Humans and animals shall be covered with sackcloth, and they shall cry mightily to God. All shall turn from their evil ways and from the violence that is in their hands. Who knows? God may relent and change his mind; he may turn from his fierce anger, so that we do not perish.” When God saw what they did, how they turned from their evil ways, God changed his mind about the calamity that he had said he would bring upon them, and he did not do it. But this was very displeasing to Jonah, and he became angry. He prayed to the Lord and said, “O Lord! Is not this what I said while I was still in my own country? That is why I fled to Tarshish at the beginning, for I knew that you are a gracious and merciful God, slow to anger, abounding in steadfast love, and relenting from punishment. And now, O Lord, please take my life from me, for it is better for me to die than to live.” And the Lord said, “Is it right for you to be angry?” Then Jonah went out of the city and sat down east of the city and made a booth for himself there. He sat under it in the shade, waiting to see what would become of the city. The Lord God appointed a bush and made it come up over Jonah, to give shade over his head, to save him from his discomfort, so Jonah was very happy about the bush. But when dawn came up the next day, God appointed a worm that attacked the bush, so that it withered. When the sun rose, God prepared a sultry east wind, and the sun beat down on the head of Jonah so that he was faint and asked that he might die. He said, “It is better for me to die than to live.” But God said to Jonah, “Is it right for you to be angry about the bush?” And he said, “Yes, angry enough to die.” Then the Lord said, “You are concerned about the bush, for which you did not labor and which you did not grow; it came into being in a night and perished in a night. And should I not be concerned about Nineveh, that great city, in which there are more than a hundred and twenty thousand persons who do not know their right hand from their left and also many animals?”
"""

# Encode the texts
llm_embedding = encode(llm_generated_text)
biblical_embedding = encode(biblical_text)

# Calculate cosine similarity
similarity = 1 - cosine(llm_embedding.numpy(), biblical_embedding.numpy())
print("Semantic similarity:", similarity)