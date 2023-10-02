This is a PCA reducing demensional and modelling case with using published data in **Numpy database**

* <p>The database is a big data include 64 colums (factors) to be consider, so key studysteps are cleaning data, reducing dimensions,different (ML) learning models building up,classified groups into 3 dimensional.
<br>
Key code:

```
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

digits = load_digits()
x_data = digits.data
y_data = digits.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.3)

nml = MLPClassifier(hidden_layer_sizes=(20,40),max_iter=500)
model = nml.fit(x_train,y_train)
