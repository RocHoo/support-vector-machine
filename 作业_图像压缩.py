
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 64
china = load_sample_image("china.jpg")
china = np.array(china, dtype=np.float64) / 255
w, h, d = original_shape = tuple(china.shape)

image_array = np.reshape(china, (w * h, d))
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)



def recreate_image(codebook, labels, w, h):
    
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


plt.figure(1)
#plt.clf()
ax = plt.axes([0, 0, 1, 1])
#plt.axis('off')
#plt.title('Original image (96,615 colors)')
plt.imshow(china)

plt.figure(2)
#plt.clf()
ax = plt.axes([0, 0, 1, 1])
#plt.axis('off')
#plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.show()
