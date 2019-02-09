import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import metrics
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches
from sklearn import preprocessing
import sys

filename = sys.argv[-1]

ALLdata = pd.read_csv(filename, header=None, skiprows=0, sep='\t')
ALLdata.columns = ['gene', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','class']
print(ALLdata.shape)

targetdata = ALLdata['class'].values.ravel()
print(targetdata.shape)
targetdata = preprocessing.label_binarize(targetdata, classes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
data = ALLdata[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']]
data = np.array(data.values)

X_train, X_test, y_train, y_test = train_test_split(data, targetdata, test_size=0.2, random_state=0)

mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(400,200), activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    predict_probabilities = clf.predict_proba(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), predict_probabilities.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='macro')
print(aus_score)
# plt.plot(fpr,tpr,label='ROC curve for 10 ANN (area = %0.4f)' % aus_score)
print(f1 / fold)





