from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as svc
from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score
from sklearn.model_selection import cross_val_score


import dbscan as db_result
import pandas as pd
import numpy as np
from matplotlib import pyplot
import operator

### doesn't use any feature extraction
label_yes_df = pd.DataFrame(db_result.label)
label_no_df = pd.DataFrame(db_result.label_no)
label_df = label_yes_df.append(label_no_df, ignore_index=True)

y_yes_df = pd.DataFrame(db_result.classification)
y_yes_df.columns = ["target"]
y_no_df = pd.DataFrame(db_result.classification_no)
y_no_df.columns = ["target"]


y_df = y_yes_df.append(y_no_df, ignore_index=True)
total_data = pd.concat([label_df, y_df], axis = 1)

index = total_data.keys()
test_size = 0.3
seed = 892
dim_red = 28


c_dict = {'0.3' : 0,
        '0.5' : 0,
        '0.7' : 0,
        '1' : 0,
        '1.2' : 0,
        '1.5' : 0,
        '2' : 0}



for i in range(100):

    X_train, X_test, y_train, y_test = train_test_split(
        total_data[index[0:(len(index)-1)]], total_data[index[len(index)-1]], test_size = test_size)


    C = [0.3, 0.5, 0.7, 1, 1.2, 1.5, 2]
    cv_scores = []

    for c in C :
        svm = svc(kernel = "rbf", probability = True, gamma='auto', C = c)
        scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())


    c = C[np.argmax(cv_scores)]
    svm = svc(kernel = "rbf", probability = True, gamma='auto', C = c)
    svm.fit(X_train, y_train.values)
    predict = np.array(svm.predict(X_test))
    acc_1 = 1 - np.sum((predict - np.array(y_test))*(predict - np.array(y_test)))/len(predict)
    predict_prob_svm = svm.predict_proba(X_test)
    auc = roc_auc_score(y_test, predict_prob_svm[:, 1])

    c_dict[str(c)] = c_dict[str(c)] + 1
    print("c select")
    print(c)
    print("accuracy")
    print(acc_1)
    print("auc")
    print(auc)



c = max(c_dict.items(), key=operator.itemgetter(1))[0]
print("final")
print(c)


svm = svc(kernel = "rbf", probability = True, gamma='auto', C = float(c))
svm.fit(X_train, y_train.values)

# performance
predict = np.array(svm.predict(X_test))
acc_1 = 1 - np.sum((predict - np.array(y_test))*(predict - np.array(y_test)))/len(predict)
predict_prob_svm = svm.predict_proba(X_test)
conf_matrix = confusion_matrix(y_true = y_test, y_pred = predict)
svm_roc = roc_curve(y_test, predict_prob_svm[:,1])
auc = roc_auc_score(y_test, predict_prob_svm[:,1])
ns_probs = [1 for _ in range(len(y_test))]
print("acc")
print(acc_1)
print("auc")
print(auc)
print("confucion matrix")
print(conf_matrix)


ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, predict_prob_svm[:,1])
pyplot.plot(ns_fpr, ns_tpr, linestyle='--')
pyplot.plot(svm_fpr, svm_tpr, marker='.', label='SVM')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


