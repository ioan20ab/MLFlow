# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:59:43 2021

@author: Johnn
"""

import sys
import mlflow.tracking
import mlflow.pyfunc
import sklearn.metrics as skm
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("dataset.json", orient="split")
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(df[['Direction', 'Speed']], df[['Total']], test_size=0.33, shuffle=False)

class FinalModel(mlflow.pyfunc.PythonModel):

    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
        from sklearn.linear_model import LinearRegression
        
        self.pipeline = Pipeline([
            ("Cat_features", ColumnTransformer([
            ("OneHot", OneHotEncoder(handle_unknown="ignore"), ["Direction"])
            ])),
            ("Poly", PolynomialFeatures(degree = 3)),
            ("LenReg", LinearRegression())
        ])

    def fit(self,x,y):
        model = self.pipeline.fit(x, y)
        self.model = model
        return self

    def predict(self, context, samples):
        return self.pipeline.predict(samples)

model = FinalModel().fit(X_train, y_train)
preds = model.predict(None, X_test)

MAE = skm.mean_absolute_error(y_test, preds)
mlflow.log_metric("MAE", MAE)
print("MAE", MAE)

r2 = skm.r2_score(y_test, preds)
mlflow.log_metric("r2", r2)
print("r2", r2)

print("Saving model")


mlflow.pyfunc.save_model("model", python_model=model, conda_env="conda.yaml")