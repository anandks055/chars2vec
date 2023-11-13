import chars2vec
import matplotlib.pyplot as plt
import sklearn.decomposition
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

dim = 50

path_to_model = 'path/to/model/directory'

X_train = [
           ('tapaH sva adhyaaya niratam', 'tapaswadhyaya niratham','tapswaadhyaaya niratham','tapaH sva adhyaaya niratam','tapaH sva adhyaaya niratam','tapas adhyayam karotu'),
           ('sita','seetham','seetam','sitha','sithaa','seeta'),
           ('ramah','ramat','raamasya','raamam','raamah','ramaya'),
           ('seetha','geetha','litha','beetha','mita','bita'),
           ('rama','soma','kama','bamah','lamah','boma'),
           ('tapaH svaadhyaaya niratam','tapasvii','vaagvidaam','varam','adhyaaya','tapas'),
           ('naaradam pari papracCha','naaradam','vaalmiikiH','muni pumgavam','naaradam pari papracCha vaalmiikiH','pumgavam'),
           ('tapaH svaadhyaaya niratam','naaradam pari papracCha vaalmiikiH','sasdnojal','gasddas','sdfsaa','beaskdnesaa')
           # not similar words, target is equal 1
          ]

y_train = [0,0,0,1,1,0,0,1]

model_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
               '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<',
               '=', '>', '?', '@', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
               'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
               'x', 'y', 'z']

sanskrit_chars = ['a','A','i','I','u','U','R^i','R^I','L^i','L^I','e','ai','o','au','aM','aH','a.N',
                  'k','kh','g','gh','~N',
                  'ch','Ch','j','jh','~n',
                  'T','Th','D','Dh','N',
                  't','th','d','dh','n',
                  'p','ph','b','bh','m',
                  'y','r','l','v',
                  'sh','Sh','s','h'
                  ]
# Create and train chars2vec model using given training data
my_c2v_model = chars2vec.train_model(dim, X_train, y_train, sanskrit_chars)

# Save pretrained model
chars2vec.save_model(my_c2v_model, path_to_model)

words = ['ramaya', 'seetaam', 'krita','seeta','sita','ramat','kreeda','tapaH svaadhyaaya niratam tapasvii vaagvidaam varam','naaradam pari papracCha vaalmiikiH muni pumgavam','tapaH svaadhyaaya niratam','vaagvidaam varam','pari papracCha','muni pumgavam']
print(len(words))
# Load pretrained model, create word embeddings
c2v_model = chars2vec.load_model(path_to_model)
word_embeddings = c2v_model.vectorize_words(words)

projection_2d = sklearn.decomposition.PCA(n_components=2).fit_transform(word_embeddings)
print(len(projection_2d))

# Draw words on plane
f = plt.figure(figsize=(8, 6))

for j in range(len(projection_2d)):
    plt.scatter(projection_2d[j, 0], projection_2d[j, 1],
                marker=('$' + words[j] + '$'),
                s=500 * len(words[j]), label=j,
                facecolors='green' if words[j]
                            in ['rama', 'seeta', 'kriya'] else 'black')

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.cm as cm

# Assuming you have already computed projection_2d using PCA and loaded the word embeddings
# projection_2d = sklearn.decomposition.PCA(n_components=2).fit_transform(word_embeddings)

# Select a specific point (for example, the first point in projection_2d)
selected_point = projection_2d[0]

# Calculate distances between the selected point and all other points
distances_to_selected_point = np.linalg.norm(projection_2d - selected_point, axis=1)

# Create a list of point indices
point_indices = list(range(len(projection_2d)))

# Get words corresponding to point_indices
words_for_indices = [words[i] for i in point_indices]

# Create a colormap
colormap = cm.get_cmap('viridis')  # You can choose a different colormap

# Create a histogram of distances to the selected point with colored bars
n_bins = 20  # You can adjust the number of bins as needed
colors = distances_to_selected_point - min(distances_to_selected_point)
colors = colors / max(colors)
plt.bar(words_for_indices, distances_to_selected_point, color=colormap(colors))
plt.xlabel('Words')
plt.ylabel('Distance to Selected Point')
plt.title('Histogram of Distances to Selected Point vs. Words in projection_2d')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility

plt.show()
