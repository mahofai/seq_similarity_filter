import pandas as pd
import os
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pyfunc
from mlflow import log_metric, log_param, log_artifact
from scipy.stats import pearsonr

from autogluon.common.features.types import R_INT,R_FLOAT,R_OBJECT,R_CATEGORY,S_TEXT_AS_CATEGORY 

from autogluon.features.generators import CategoryFeatureGenerator, AsTypeFeatureGenerator, BulkFeatureGenerator, DropUniqueFeatureGenerator, FillNaFeatureGenerator, PipelineFeatureGenerator, OneHotEncoderFeatureGenerator,IdentityFeatureGenerator

from feature_generator import one_hot_Generator



train_feature_generator = PipelineFeatureGenerator(
    generators=[
        [   one_hot_Generator(verbosity=3,features_in=['seq'],seq_type = "protein"),
            IdentityFeatureGenerator(infer_features_in_args=dict(
                valid_raw_types=[R_INT, R_FLOAT])),
        ],
        
     ],
    verbosity=3,
    post_drop_duplicates=False,
    post_generators=[IdentityFeatureGenerator()]
)


class AutogluonModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor):
        self.predictor = predictor
        
    # def load_context(self, context):
    #     print("Loading context")
    #     # self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))
    #     self.predictor = context.artifacts.get("predictor_path")

    def predict(self, model_input):
        return self.predictor.predict(model_input)
    
    def evaluate(self, model_input):
        return self.predictor.evaluate(model_input)
    
    def leaderboard(self, model_input):
        return self.predictor.leaderboard(model_input)


if __name__ == "__main__":
    train_data = TabularDataset('train.csv')
    test_data = TabularDataset('test.csv')
    train_data = train_data.iloc[:1000]
    test_data = test_data.iloc[:100]
    concatenated_df  = pd.concat([train_data,test_data], axis=0)
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    print("alpha",alpha)
    print("l1_ratio",l1_ratio)
    
    train_feature_generator = PipelineFeatureGenerator(
        generators=[
            [   one_hot_Generator(verbosity=3,features_in=['seq'],seq_type = "protein"),
                IdentityFeatureGenerator(infer_features_in_args=dict(
                    valid_raw_types=[R_INT, R_FLOAT])),
            ],
        ],
        verbosity=3,
        post_drop_duplicates=False,
        post_generators=[IdentityFeatureGenerator()]
    )
    one_hot_all_data = train_feature_generator.fit_transform(X=concatenated_df)
    
    one_hot_train_data = one_hot_all_data[:len(train_data)]
    one_hot_test_data = one_hot_all_data[len(train_data):]
    
    # print(concatenated_df)
    one_hot_valid_data1 = one_hot_train_data[one_hot_train_data["fold"] ==0.0]
    one_hot_train_data1 = one_hot_train_data[one_hot_train_data["fold"] !=0.0]

    one_hot_train_data1 = one_hot_train_data1.drop(["fold"],axis=1)
    one_hot_valid_data1 = one_hot_valid_data1.drop(["fold"],axis=1)
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 1
    print("alphaP:",alpha)
    print("time_limit:",time_limit)

    print("one_hot_train_data1.shape",one_hot_train_data1.shape)
    # print(one_hot_valid_data1)

    with mlflow.start_run() as run:
        
        # tracking_uri = "http://127.0.0.1:5001"
        # mlflow.set_tracking_uri(tracking_uri)
        # print(mlflow.get_tracking_uri())

        predictor = TabularPredictor(label='solubility',eval_metric="precision")
        predictor.fit(train_data=one_hot_train_data1, tuning_data=one_hot_valid_data1, feature_generator=None, time_limit=time_limit)
        
        evaluation = predictor.evaluate(one_hot_test_data, silent=True)
        print("test eval:",evaluation)
        
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("precision", evaluation["precision"])
        mlflow.log_metric("auc", evaluation["roc_auc"])
        mlflow.log_metric("mcc", evaluation["precision"])

        # predictor.leaderboard(one_hot_test_data, silent=True)
        
        # model = AutogluonModel(predictor)
        
        # model_info = mlflow.pyfunc.log_model(
        #     artifact_path='ag-model', python_model=model
        # )
        
        
        # predictor.evaluate(one_hot_test_data)
        
        # predictor.evaluate(one_hot_valid_data1)
        
        ## evaluate mode demo
        # val_metrics = loaded_model.evaluate(one_hot_valid_data1)
        # test_metrics = loaded_model.evaluate(one_hot_test_data)
        # for k,v in val_metrics.items():
        #     log_metric('valid_'+k, v)
        # for k,v in test_metrics.items():
        #     log_metric('test_'+k, v)
            
        mlflow.end_run()
