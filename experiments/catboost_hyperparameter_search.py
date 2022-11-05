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

import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from catboost import CatBoostRegressor, Pool

from preprocessor import DataPreprocessorV2
from helpers import calculate_metrics, plot_scatterplot, plot_distributions2

mlflc = MLflowCallback(metric_name=["log_mae_int", "r2_int"],
                       mlflow_kwargs={"experiment_id": "5"})


@mlflc.track_in_mlflow()
def objective(trial: optuna.Trial) -> float:
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
        "has_time":
        True,
        "random_seed":
        42,
        "task_type":
        "GPU",
        "iterations":
        1000,
        "loss_function":
        "RMSE",
        "depth":
        trial.suggest_int("depth", 8, 16),
        "l2_leaf_reg":
        trial.suggest_float('l2_leaf_reg', 1.0, 5.5, step=0.5),
        "border_count":
        trial.suggest_categorical("border_count", [128, 254]),
        "grow_policy":
        trial.suggest_categorical('grow_policy',
                                  ["SymmetricTree", "Depthwise"]),
        "bootstrap_type":
        trial.suggest_categorical("bootstrap_type",
                                  ["Bayesian", "Bernoulli", "MVS"]),
    }

    if train_params["bootstrap_type"] == "Bayesian":
        train_params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0, 10)
    elif train_params["bootstrap_type"] == "Bernoulli":
        train_params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    # Create model
    print("Training model...")
    model = CatBoostRegressor(**train_params)
    model.fit(train_pool, logging_level="Silent")

    # evaluate model
    print("Evaluating model...")
    predicted = model.predict(test_pool)
    predicted_unscaled = preprocessor.unscale_price(predicted.reshape(-1, 1))
    y_true_unscaled = df_test["Price"].values.reshape(-1, 1)

    # save model artifact
    feature_importances = dict(
        zip(df_train_p["columns"], model.feature_importances_))
    mlflow.log_text(json.dumps(feature_importances),
                    "feature_importances.json")
    mlflow.log_text(json.dumps(model.get_all_params()), "model_params.json")

    # save plots
    fig_scatter = plot_scatterplot(y_true_unscaled, predicted_unscaled)
    mlflow.log_figure(fig_scatter, "scatterplot.png")
    plt.close(fig_scatter)

    fig_scatter = plot_scatterplot(y_true_unscaled,
                                   predicted_unscaled,
                                   logy=True)
    mlflow.log_figure(fig_scatter, "scatterplot-log.png")
    plt.close(fig_scatter)

    fig_dist = plot_distributions2(y_true_unscaled, predicted_unscaled)
    mlflow.log_figure(fig_dist, "distributions.png")
    plt.close(fig_dist)

    # calculate metrics
    reg_metrics = calculate_metrics(y_true_unscaled, predicted_unscaled)
    mlflow.log_metrics(reg_metrics)

    # return log mae
    return reg_metrics["log_mae"], reg_metrics["r2"]


if __name__ == "__main__":
    study = optuna.create_study(study_name="catboost_hyperparam2",
                                directions=["minimize", "maximize"],
                                storage="sqlite:///catboost_hyperparam2.db",
                                load_if_exists=True)
    print(f"Sampler is {study.sampler.__class__.__name__}")

    study.optimize(objective,
                   n_trials=50,
                   timeout=None,
                   callbacks=[mlflc],
                   n_jobs=1)

    print("Number of finished trials: {}".format(len(study.trials)))
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    optuna.visualization.plot_pareto_front(study,
                                           target_names=["FLOPS", "accuracy"])

    optuna.visualization.plot_param_importances(study,
                                                target=lambda t: t.values[0],
                                                target_name="MAE")
