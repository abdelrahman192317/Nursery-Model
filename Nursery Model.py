import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

########################################################################

#Importing the dataset
Data = pd.read_csv('E://projects//Python//Machine Learning Project//Drug//nursery.data', names=[0,1,2,3,4,5,6,7,8])

X_data = Data.iloc[:,:-1].values
Y_data = Data.iloc[:,-1].values

########################################################################

#preprocessing
from sklearn.compose import ColumnTransformer
oh = ColumnTransformer([('encoder',OneHotEncoder(),[0,1,2,3,4,5,6])],remainder="passthrough")
X_data = np.array(oh.fit_transform(X_data))
oh = ColumnTransformer([('encoder',OneHotEncoder(),[-1])],remainder="passthrough")
X_data = np.array(oh.fit_transform(X_data))

'''
le = LabelEncoder()
X_data[:,0] = le.fit_transform(X_data[:,0])
X_data[:,-1] = le.fit_transform(X_data[:,-1])

sc=MinMaxScaler()
X_data[:,1:]=sc.fit_transform(X_data[:,1:])

########################################################################
'''
#Splitting the dataset into the Training set, validation and Test set
X, x_test, Y, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=0)
x_train, x_vali, y_train, y_vali = train_test_split(X, Y, test_size=0.33, random_state=42)


########################################################################

#Training the K-NN model on the Training set
#from sklearn.neighbors import KNeighborsClassifier
#tree = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

#Training the Naive Bayes model on the Training set
# from sklearn.naive_bayes import GaussianNB
# tree = GaussianNB()

#Training the Decision Tree model on the Training set
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=15, criterion='entropy', random_state=2)

#Training the Random Forest Classifier model on the Training set
#from sklearn.ensemble import RandomForestClassifier
#tree = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', random_state = 0)

tree.fit(x_train, y_train)

########################################################################

# Predicting the Test set results
y_pred_train = tree.predict(x_train)
print(accuracy_score(y_train, y_pred_train))

y_pred_vali = tree.predict(x_vali)
print(accuracy_score(y_vali, y_pred_vali))

y_pred_test = tree.predict(x_test)
print(accuracy_score(y_test, y_pred_test))

########################################################################

max_depth_values=[]
for i in range(10,20):
    max_depth_values.append(i)
train_acc_val=[]
test_acc_val=[]
vali_acc_val=[]
for max_depth_val in max_depth_values:
    tree = tree = DecisionTreeClassifier(max_depth=max_depth_val, criterion='entropy', random_state=2)
    #print(max_depth_val)
    tree.fit(x_train, y_train)
    y_pred_train = tree.predict(x_train)
    y_pred_vali = tree.predict(x_vali)
    y_pred_test = tree.predict(x_test)
    train_acc_val.append(accuracy_score(y_train, y_pred_train))
    vali_acc_val.append(accuracy_score(y_vali, y_pred_vali))
    test_acc_val.append(accuracy_score(y_test, y_pred_test))

    
import matplotlib.pyplot as plt
plt.plot(max_depth_values, train_acc_val, label='acc train')
plt.plot(max_depth_values, vali_acc_val, label='acc vali')
plt.plot(max_depth_values, test_acc_val, label='acc test')
plt.legend()
plt.grid(axis='both')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.title('Effect of Max depth on accuracy')
plt.show()
