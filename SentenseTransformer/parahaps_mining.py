from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining

model = SentenceTransformer("all-MiniLM-L6-v2")

# Single list of sentences - Possible tens of thousands of sentences
sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
]

paraphrases = paraphrase_mining(model, sentences)
print("paraphrase : ")
print(paraphrases)
print("")

print("{:35} {:35} Score".format("sentence[i]", "sentences[j]"))
for paraphrase in paraphrases[0:10]:
    score, i, j = paraphrase
    print("{:35} {:35} {:.4f}".format(sentences[i], sentences[j], score))