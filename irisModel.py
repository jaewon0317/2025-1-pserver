import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width:float

class IrisMachineLearningModel:
    def __inin__(self):
        self.iris_df = pd.read_csv('iris.csv')
        self.rfc_fname='iris_rfc.pkl'
        try:
            self.model_rfc = joblib.load(self.rfc_fname)
        except Exception as _:
            self.model_rfc = self.rfc_train()
            joblib.dump(self.model_rfc, self.rfc_fname)
        return

    def rfc_train(self):
        X = self.rfc_fname.drop('species', axis=1)
        y = self.rfc_fname['species']
        rfc = RandomForestClassifier()
        model = rfc.fit(X,y)
        return model
    def predict_species(self, sepal_lenghth,spepa_width,petal_length,petal_width):
        X_new = np.array([[sepal_lenghth,spepa_width,petal_length,petal_width]])
        prediction = self.model_rfc.predict(X_new)
        print(prediction)
        return prediction[0]