# 2025.3.10
# 프로젝트2 붓꽃분류기 만들기
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

iris_df = pd.read_csv('iris.csv')
y = iris_df['species']
X = iris_df.drop('species',axis=1)

kn = KNeighborsClassifier()
rfc = RandomForestClassifier()

model_kn = kn.fit(X,y)
model_rfc = kn.fit(X,y)

joblib.dump(model_rfc, 'model_rfc.pkl')

X_new = np.array([[5.0,3.4,1.4,2.1]])

model_rfc = joblib.load('model_rfc.pkl')

prediction = model_kn.predict(X_new)
prediction = model_rfc.predict(X_new)
print(prediction)
probability = model_kn.predict_proba(X_new)
probability = model_rfc.predict_proba(X_new)
print(probability)
