from gensim.models import Word2Vec
from machinelearning_practical.src.preprocessing.text_preprocessing import load_text
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

# tweets = load_text("../../data/train_text_sample.txt")
tweets = load_text("../../data/train_text.txt")
print(tweets[0])

# train model
model = Word2Vec(tweets, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.key_to_index)
print(words)
# access vector for one word
print(model.wv['love'])
word_embeddings = model.wv[model.wv.key_to_index]
# # save model
# model.save('model.bin')
# # load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)

tsne = TSNE(random_state=0)
data = tsne.fit_transform(word_embeddings)
df = pd.DataFrame(data=data, columns=["Dimension_1", "Dimension_2"])

plt.figure(figsize=(10, 6))
plt.scatter(df["Dimension_1"], df["Dimension_2"], alpha=1)
plt.title('t-SNE Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
