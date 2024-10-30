# -*- coding: utf-8 -*-
"""

@author: Johnn
"""
import pandas as pd
import mlflow


# pip azureml (pip install azureml-core):
from azureml import core
from azureml.core import Workspace
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())



# mlflow.set_tracking_uri("http://training.itu.dk:5000/")

# Set the experiment name
mlflow.set_experiment("ML_flow")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="degree1"):
    # TODO: Insert path to dataset
    df = pd.read_json("dataset.json", orient="split")
    df.head()
    print(df.columns)
    # TODO: Handle missing data
    df = df.dropna()
    
    degree = 3
    pipeline = Pipeline([
        # TODO: You can start with your pipeline from assignment 1
        ("Cat_features", ColumnTransformer([
            ("OneHot", OneHotEncoder(handle_unknown="ignore"), ["Direction"])
        ])),
        ("Poly", PolynomialFeatures(degree = degree)),
        ("LenReg", LinearRegression())
    ])

    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    metrics = [
        ("MAE", mean_absolute_error, []),
    ]

    X = df[["Speed","Direction"]]
    y = df["Total"]

    number_of_splits = 3

    #TODO: Log your parameters. What parameters are important to log?
    #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    
    mlflow.log_param('Nsplits', number_of_splits)
    mlflow.log_param('degree', degree)
    mlflow.log_param('model', pipeline.steps[-1][0])
    
    
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]
        
        """
        import matplotlib
        matplotlib.use('agg')
        from matplotlib import pyplot as plt
        plt.plot(truth.index, truth.values, label ="Truth")
        plt.plot(truth.index, predictions , label ="Predictions")
        plt.show()
        """

        
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
    
    # Log a summary of the metrics
    for name, _, scores in metrics:
        # NOTE: Here we just log the mean of the scores. 
        # Are there other summarizations that could be interesting?
        mean_score = sum(scores)/number_of_splits
        mlflow.log_metric(f"mean_{name}", mean_score)
