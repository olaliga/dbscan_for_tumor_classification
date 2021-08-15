import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing as pre
import keyboard as kb
import pandas as pd

#### 調: cv2.COLOR_BGR2LAB,


def detect(img_path, ratio):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) ## 調

    ## 降維度
    n = 0
    while (n < 3):  ### 降 4 次 可調
        labimg = cv2.pyrDown(labimg)  ### 調 他具有降維效果
        n = n + 1

    feature_image = np.reshape(labimg, [-1, 3])


    rows, cols, chs = labimg.shape

    minpts = max(2, int(feature_image.shape[0] * ratio))

    neigh = NearestNeighbors(n_neighbors=minpts)
    nbrs = neigh.fit(feature_image)
    distances, indices = nbrs.kneighbors(feature_image)
    distances = np.sort(distances, axis=0)
    distances = distances[:, -1]
    eps = min(max(1,np.unique(distances)[-1]),3)  ##### 改
    print(eps)

    ## 10 5
    db = DBSCAN(eps=eps, min_samples=minpts, metric='euclidean', algorithm='auto')  ### 調 他具有降維效果
    db.fit(feature_image)
    labels = db.labels_

    need = [eps, labels]

    return need



def detect_img(img_path, ratio):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) ## 調

    ## 降維度
    n = 0
    while (n < 3):  ### 降 4 次 可調
        labimg = cv2.pyrDown(labimg)  ### 調 他具有降維效果
        n = n + 1

    feature_image = np.reshape(labimg, [-1, 3])


    rows, cols, chs = labimg.shape
    print("rows")
    print(rows)
    print("cols")
    print(cols)
    minpts = max(2, int(feature_image.shape[0] * ratio))

    neigh = NearestNeighbors(n_neighbors=minpts)
    nbrs = neigh.fit(feature_image)
    distances, indices = nbrs.kneighbors(feature_image)
    distances = np.sort(distances, axis=0)
    distances = distances[:, -1]
    eps = min(max(1,np.unique(distances)[-1]),3)  ##### 改
    print(eps)

    ## 10 5
    db = DBSCAN(eps=eps, min_samples=minpts, metric='euclidean', algorithm='auto')  ### 調 他具有降維效果
    db.fit(feature_image)
    labels = db.labels_

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(labimg, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(np.reshape(labels, [rows, cols]))
    plt.axis('off')

    '''
    plt.subplot(2, 2, 4)
    plt.imshow(labimg2)
    plt.axis('off')
    '''

    plt.show()

    need = [eps, labels]

    return need


### need
label = []
eps = []
classification = []
unique = []
count = []

###
current_dir = os.getcwd()
os.chdir('tumor/yes')
image_list = os.listdir()

for i in image_list:
    result_eps, result_labels = detect(i, ratio = 0.025)
    label.append(result_labels)
    eps.append(result_eps)


for i in range(len(label)):
    classification.append(1)
    uni_dum, count_dum = np.unique(label[i], return_counts = True)
    unique.append(uni_dum)
    count.append(count_dum)

print("no")

os.chdir(current_dir)
os.chdir('tumor/no')
image_list = os.listdir()

### need
label_no = []
eps_no = []
classification_no = []
unique_no = []
count_no = []


for i in image_list:
    classification_no.append(0)
    result_eps, result_labels = detect(i, ratio = 0.025)
    label_no.append(result_labels)
    eps_no.append(result_eps)

###

label_yes_df = pd.DataFrame(label)
label_no_df = pd.DataFrame(label_no)
label_df = label_yes_df.append(label_no_df, ignore_index=True)

y_yes_df = pd.DataFrame(classification)
y_no_df = pd.DataFrame(classification_no)
y_df = y_yes_df.append(y_no_df, ignore_index=True)






if __name__ == '__main__':
    for i in range(len(image_list)):
        uni_dum, count_dum = np.unique(label[i], return_counts = True)
        unique_no.append(uni_dum)
        count_no.append(count_dum)

    scale_count = []
    for i in range(len(count)):
        need = count[i]/np.sum(count[i])
        scale_count.append(need)

    scale_count_no = []
    for i in range(len(count_no)):
        need = count_no[i]/np.sum(count_no[i])
        scale_count_no.append(need)





    pd_scale_count = pd.DataFrame(scale_count)
    g1 = (pd_scale_count[0], 1-pd_scale_count[0])

    pd_scale_count_no = pd.DataFrame(scale_count_no)
    g2 = (pd_scale_count_no[0], 1-pd_scale_count_no[0])

    colors = ("red", "green")
    groups = ("Yes", "No")


    data_plot = pd.DataFrame(pd_scale_count[0])
    data_plot = data_plot.assign(group = np.repeat("Yes", len(pd_scale_count[0])))

    df2 = pd.DataFrame(pd_scale_count_no[0])
    df2 = df2.assign(group = np.repeat("No", len(pd_scale_count_no[0])))

    data_plot = data_plot.append(df2, ignore_index=True)


    ax = sns.boxplot(y=0, x="group", data=data_plot)









    os.chdir(current_dir)
    os.chdir('tumor/no')
    image_list = os.listdir()
    result_eps, result_labels = detect_img(image_list[30], 0.025)
    uni_dum, count_dum = np.unique(result_labels, return_counts = True)
    print(result_labels.shape)
    ratio = count_dum/np.sum(count_dum)

    # no : 3, 4, 5 noise 變成不是 tumor,
