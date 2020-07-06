import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

vector_size = 50
embeddings_dict = {}
with open("glove/glove_korean_sns_50d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            vector = np.asarray(values[1:], "float32")
            if len(vector) == vector_size:
                embeddings_dict[word] = vector
            else:
                print(str(len(vector)))

        except:
            print('not float... ' + word)


def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

print(find_closest_embeddings(embeddings_dict["이재명"])[1:6])

print(find_closest_embeddings(
    embeddings_dict["이재명"] - embeddings_dict["정치인"] + embeddings_dict["남경필"]
)[:5])

tsne = TSNE(n_components=2, random_state=0)
words =  list(embeddings_dict.keys())
vectors = [embeddings_dict[word] for word in words]
Y = tsne.fit_transform(vectors[:1000])

plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()