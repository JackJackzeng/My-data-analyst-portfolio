This is a fluit classiy case to be used into farm (AI intellegent to classify the fluits). I studied and modelling the training data,built a **KNN model** and classified it using **K_Neighbors_Classifier** method.
* **K-NN Varying the number of neighbors** to justify the right **K**.
* **Mplot3d**
  to solve this problem.

* The database is a published data include 5 colums (factors) to be consider, so key steps are cleaning data, (ML) learning models building up,justify the proper **K** value and classified groups.
<br>
Key code:

```
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
