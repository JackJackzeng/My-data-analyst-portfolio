This is a case study using python /NumPy/ Pandas/Matplot tools to train the **titanic data** learning using different mathematic algorithms. The data source can be downlaod from the public website, such as [Titanic dataset](https://www.kaggle.com/datasets/hesh97/titanicdataset-traincsv)

key code :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('titanic_train.csv')
data.shape

data.describe()

data['Age'] = data['Age'].fillna(data['Age'].median())
data

from sklearn.preprocessing import LabelEncoder
LD = LabelEncoder()
data['Sex'] = LD.fit_transform(data['Sex'])

print(data['Embarked'].unique())
data['Embarked'] = data['Embarked'].fillna('S')

LD = LabelEncoder()
data['Embarked'] = LD.fit_transform(data['Embarked'])
print(data['Embarked'].unique())

x_columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'] 
x_data = data[x_columns]
y_data = data['Survived']

x_data.head()

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
x_data = SS.fit_transform(x_data)
x_data

from sklearn import tree
dtree = tree.DecisionTreeClassifier(max_depth=5,min_samples_split=5,min_samples_leaf=5)
model = dtree.fit(x_data,y_data)
print(model.score(x_data,y_data))

from sklearn.ensemble import RandomForestClassifier
RF1 = RandomForestClassifier(random_state=10,min_samples_split=5,min_samples_leaf=5,n_estimators=80)
model = RF1.fit(x_data,y_data)

print(model.score(x_data,y_data))

```

If you are interested reviewing and testing the project please use my [**Google CoLab notebook**](https://colab.research.google.com/drive/1FhjdinLX9dejz4spkXrhXDHuENXQM3Fq#updateTitle=true&folderId=1Q9EqShSEW9F3ULWA9Z6sSSbFlLBSQTmO)

[Click here to return to the main page](../README.md)
