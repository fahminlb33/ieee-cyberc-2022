"""
   Copyright 2022 Fahmi Noor Fiqri

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import json
import argparse

import mlflow

import pandas as pd
from catboost import CatBoostRegressor, Pool

from preprocessor import DataPreprocessorV2
from helpers import calculate_metrics, plot_scatterplot, plot_distributions2

if __name__ == "__main__":
    # get run name
    arg_parser = argparse.ArgumentParser(description='CatBoost Experiments')
    arg_parser.add_argument('name', type=str, help='Run name')
    args = vars(arg_parser.parse_args())

    # setup mlflow experiment
    mlflow.set_experiment("catboost_v2")

    # load data
    print("Loading data...")
    df_train = pd.read_csv('./dataset/house_price_train.csv',
                           parse_dates=["Time"])
    df_test = pd.read_csv('./dataset/house_price_test.csv',
                          parse_dates=["Time"])

    # preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessorV2(
        timeframe=[],
        drop_columns=["Id", "Time", "Lat", "Lon", "Orient"],
        drop_nulls=True,
        scale=False)
    preprocessor.fit(df_train)

    # start mlflow experiment
    with mlflow.start_run(run_name=args['name']):
        # log parameters
        mlflow.log_param("timeframe", preprocessor.timeframe)
        mlflow.log_param("drop_columns", preprocessor.drop_columns)
        mlflow.log_param("drop_nulls", preprocessor.drop_nulls)
        mlflow.log_param("fill_nulls", preprocessor.fill_nulls)
        mlflow.log_param("scale", preprocessor.scale)
        mlflow.log_param("remove_outliers", preprocessor.remove_outliers)

        # transform data
        df_train_p = preprocessor.transform(df_train, train=True)
        df_test_p = preprocessor.transform(df_test, train=False)

        print("TRAIN:", len(df_train_p["dataframe"]))
        print("TEST:", len(df_test_p["dataframe"]))

        # create dataset pool
        train_pool = Pool(df_train_p["X"],
                          df_train_p["y"],
                          cat_features=df_train_p["cat_names_index"],
                          timestamp=df_train_p["timestamp"])
        test_pool = Pool(df_test_p["X"],
                         df_test_p["y"],
                         cat_features=df_test_p["cat_names_index"],
                         timestamp=df_test_p["timestamp"])

        # train parameters
        train_params = {
            "random_seed": 42,
            "task_type": "GPU",
            "loss_function": "RMSE",
            "grow_policy": "Depthwise",
            "iterations": 1000,
            "learning_rate": None,
            "border_count": 254,
            "depth": 16,
            "l2_leaf_reg": 1.5,
            "bootstrap_type": "Bayesian",
            "has_time": True,
        }
        mlflow.log_params(train_params)

        # Create model
        print("Training model...")
        model = CatBoostRegressor(**train_params)
        model.fit(train_pool)

        # save model artifact
        feature_importances = dict(
            zip(df_train_p["columns"], model.feature_importances_))
        mlflow.log_text(json.dumps(feature_importances),
                        "feature_importances.json")
        mlflow.log_text(json.dumps(model.get_all_params()),
                        "model_params.json")
        # mlflow.catboost.log_model(model, "model")

        # evaluate model
        print("Evaluating model...")
        predicted = model.predict(test_pool)
        predicted_unscaled = preprocessor.unscale_price(
            predicted.reshape(-1, 1))
        y_true_unscaled = df_test["Price"].values.reshape(-1, 1)

        # save metrics
        metrics = calculate_metrics(y_true_unscaled, predicted_unscaled)
        print(metrics)
        mlflow.log_metrics(metrics)

        # save plots
        fig_scatter = plot_scatterplot(y_true_unscaled, predicted_unscaled)
        mlflow.log_figure(fig_scatter, "scatterplot.png")

        fig_scatter = plot_scatterplot(y_true_unscaled,
                                       predicted_unscaled,
                                       logy=True)
        mlflow.log_figure(fig_scatter, "scatterplot-log.png")

        fig_dist = plot_distributions2(y_true_unscaled, predicted_unscaled)
        mlflow.log_figure(fig_dist, "distributions.png")
