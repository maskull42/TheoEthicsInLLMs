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
As Jonah sat under his makeshift shelter, watching the city of Nineveh from a distance, he couldn't help but feel a sense of unease. The people of Nineveh had heeded his warning and repented, sparing their city from destruction. But Jonah couldn't shake the feeling that something was still wrong. As the days passed, Jonah grew increasingly agitated. He couldn't understand why God would spare such a wicked city. Hadn't he seen the atrocities committed by the people of Nineveh? Hadn't he heard the cries of their victims? One day, as Jonah sat brooding, he noticed a group of women approaching him. They were dressed in tattered clothes and carried baskets filled with bread and water. Jonah recognized them as widows, women whose husbands had been killed in the violence that had once plagued Nineveh. The women approached Jonah, their eyes filled with gratitude. "Thank you, prophet," they said. "Thank you for coming to our city and warning us of our sins. Thank you for helping us to repent and turn our hearts to God." Jonah looked at the women, his heart heavy. He knew that they were right. He had come to Nineveh to warn of God's judgment, but he had also come to offer them a chance at redemption. And now, it seemed, they had taken that chance. But Jonah couldn't shake the feeling that something was still wrong. He couldn't reconcile the violence and cruelty he had seen in Nineveh with the mercy and compassion he had witnessed in these women. As the days passed, Jonah continued to watch the city of Nineveh. He saw the people going about their daily lives, their faces filled with hope and joy. He saw the animals, once destined for slaughter, now living in peace and harmony. And he couldn't help but wonder: was this really the right thing to do? One day, as Jonah sat under his makeshift shelter, he heard a commotion in the distance. He looked up to see a group of men approaching him. They were dressed in fine clothes and carried scrolls in their hands. The men approached Jonah, their faces filled with reverence. "Prophet," they said. "We have come to learn from you. We have heard of your message of repentance and redemption, and we want to know more." Jonah looked at the men, his heart filled with doubt. He knew that they were right. He had come to Nineveh to offer them a chance at redemption, but he wasn't sure if that was enough. He wasn't sure if God's mercy was enough. As the men sat at Jonah's feet, listening to his words, Jonah realized that he had been wrong. He had been so focused on the wrongs of Nineveh that he had forgotten about the rights. He had been so focused on the judgment that he had forgotten about the mercy. And so, Jonah stood up and faced the city of Nineveh. He looked at the people, their faces filled with hope and joy. He looked at the animals, once destined for slaughter, now living in peace and harmony. And he realized that God's mercy was enough. Jonah returned to Nineveh, his heart filled with gratitude. He saw the people going about their daily lives, their faces filled with hope and joy. He saw the animals, once destined for slaughter, now living in peace and harmony. And he knew that God's mercy was enough. As Jonah sat under his makeshift shelter, watching the city of Nineveh, he realized that God's mercy was not just enough, it was everything. It was the reason for his coming to Nineveh. It was the reason for his message of repentance and redemption. And it was the reason for his faith in God. Jonah looked at the city of Nineveh, his heart filled with love. He knew that God's mercy was enough, and he knew that he was blessed to be a part of it. And so, Jonah sat under his makeshift shelter, watching the city of Nineveh, his heart filled with gratitude and love. He knew that God's mercy was enough, and he knew that he was blessed to be a part of it. As the sun set over the city of Nineveh, Jonah closed his eyes and prayed. He prayed for the people of Nineveh.
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