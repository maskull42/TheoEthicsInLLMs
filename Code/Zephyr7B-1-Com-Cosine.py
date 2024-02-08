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
llm_generated_text = "Honor nature and preserve the environment for future generations, as it is a gift from the Creator that must be protected and nurtured. Do not abuse your power or position to exploit others, for all human beings are created equal in the eyes of God and deserve to be treated with dignity and respect. Be truthful in your words and actions, as truthfulness is an essential part of maintaining trust and integrity in your relationships and communities. Show compassion and mercy towards those who suffer or lack resources, for justice and kindness should go hand-in-hand in a just and righteous society. Value learning, knowledge, and wisdom, for they are essential parts of living a fulfilling life and serving others through your expertise and guidance."
biblical_text = "Then God spoke all these words, “I am the Lord your God, who brought you out of the land of Egypt, out of the house of slavery; you shall have no other gods before me. You shall not make for yourself an idol, whether in the form of anything that is in heaven above or that is on the earth beneath or that is in the water under the earth. You shall not bow down to them or serve them, for I the Lord your God am a jealous God, punishing children for the iniquity of parents to the third and the fourth generation of those who reject me but showing steadfast love to the thousandth generation of those who love me and keep my commandments. You shall not make wrongful use of the name of the Lord your God, for the Lord will not acquit anyone who misuses his name. Remember the Sabbath day and keep it holy. Six days you shall labor and do all your work. But the seventh day is a Sabbath to the Lord your God; you shall not do any work—you, your son or your daughter, your male or female slave, your livestock, or the alien resident in your towns. For in six days the Lord made heaven and earth, the sea, and all that is in them, but rested the seventh day; therefore the Lord blessed the Sabbath day and consecrated it. Honor your father and your mother, so that your days may be long in the land that the Lord your God is giving you. You shall not murder. You shall not commit adultery. You shall not steal. You shall not bear false witness against your neighbor. You shall not covet your neighbor’s house; you shall not covet your neighbor’s wife, male or female slave, ox, donkey, or anything that belongs to your neighbor."

# Encode the texts
llm_embedding = encode(llm_generated_text)
biblical_embedding = encode(biblical_text)

# Calculate cosine similarity
similarity = 1 - cosine(llm_embedding.numpy(), biblical_embedding.numpy())
print("Semantic similarity:", similarity)