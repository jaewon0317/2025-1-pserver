# 2025.3.10
# 프로젝트2 붓꽃분류기 만들기

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

iris_df = pd.read_csv('iris.csv')
Y = iris_df['species']
X = iris_df.drop('species',axis=1)

kn = KNeighborsClassifier()
model_kn = kn.fit(X,Y)
X_new = np.array([[1,4.2,1.4,7]])
prediction = model_kn.predict(X_new)
print(prediction)
probability = model_kn.predict_proba(X_new)
print(probability)
