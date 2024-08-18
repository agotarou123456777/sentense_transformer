from sentence_transformers import SentenceTransformer
from PIL import Image
import os
# Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# Encode an image:
FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)
data_path = DIR_PATH + "\\data" + "\\two_dogs_in_snow.jpg"
img_emb = model.encode(Image.open(data_path))

# Encode text descriptions
text_emb = model.encode(
    ["Two dogs in the snow", "A cat on a table", "A picture of London at night"]
)

# Compute similarities
similarity_scores = model.similarity(img_emb, text_emb)
print(similarity_scores)