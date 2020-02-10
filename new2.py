import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def load_pics(p, pics, target = None, ttag = None):
    images = os.listdir(p)
    images
    for image in images:
        print(image.lower())
        cur_img = mpimg.imread(p+image)
        cur_img.size
        cur_img1 = cur_img.reshape(1,(120*120*3))
        pics = np.append(pics, cur_img1, axis=0)
        plt.imshow(cur_img)
        print(cur_img.shape, type(cur_img))
        if target!=None:
            target.append(ttag)
    return pics, target

tgt = []
pics = np.empty(shape=(0,120*120*3))

pics,tgt = load_pics("C:\\Users\\PC\\Desktop\\MariGold\\Scalled\\", pics, tgt, 'MariGold')
pics,tgt = load_pics("C:\\Users\\PC\\Desktop\\NonMariGold\\Scalled\\", pics, tgt, 'NonMariGold')
print(tgt)

print(pics.shape)
dataset = datasets.base.Bunch(data=pics,target = tgt)
print(dataset)

n_observations, n_features = dataset.data.shape
x = dataset.data
y = dataset.target
print(x)
print(dataset.target)

