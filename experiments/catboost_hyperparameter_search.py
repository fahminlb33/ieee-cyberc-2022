import os

import pandas as pd

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from catboost import CatBoostRegressor, Pool

from helpers import calculate_metrics, plot_scatterplot, plot_distributions
from preprocessor import DataPreprocessor

mlflc = MLflowCallback(metric_name=["log_mae_int", "r2_int"],
                       mlflow_kwargs={"experiment_id": "3"})


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
    preprocessor = DataPreprocessor(
        timeframe=[],
        drop_columns=["Id", "Lat", "Lon", "Orient"],
        drop_nulls=True,
        scale=True,
        remove_outliers=None,
        moving_average=None,
        moving_average_groupby=None)
    preprocessor.fit(df_train, df_test)

    # create dataset pool
    train_pool = Pool(preprocessor.X_train,
                      preprocessor.y_train,
                      cat_features=preprocessor.categorical_column_indexes)
    test_pool = Pool(preprocessor.X_test,
                     preprocessor.y_test,
                     cat_features=preprocessor.categorical_column_indexes)

    # train parameters
    train_params = {
        "random_seed":
        42,
        "task_type":
        "GPU",
        "loss_function":
        "RMSE",  # RMSE, Tweedie:variance_power=1.9, Poisson
        "iterations":
        1000,  # 1000
        "grow_policy":
        trial.suggest_categorical('grow_policy',
                                  ["SymmetricTree", "Depthwise"]),
        "l2_leaf_reg":
        trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5),
        "depth":
        trial.suggest_int("depth", 6, 15),
        "boosting_type":
        "Plain",
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
    model.fit(train_pool)

    # evaluate model
    print("Evaluating model...")
    predicted = model.predict(test_pool)
    predicted_unscaled = preprocessor.inverse_transform_price(
        predicted.reshape(-1, 1))
    y_true_unscaled = preprocessor.inverse_transform_price(
        preprocessor.y_test.reshape(-1, 1))

    # save plots
    fig_scatter = plot_scatterplot(y_true_unscaled, predicted_unscaled)
    mlflow.log_figure(fig_scatter, "scatterplot.png")

    fig_scatter = plot_scatterplot(y_true_unscaled,
                                   predicted_unscaled,
                                   logy=True)
    mlflow.log_figure(fig_scatter, "scatterplot-log.png")

    train_prices = preprocessor.inverse_transform_price(
        preprocessor.y_train.reshape(-1, 1))
    fig_dist = plot_distributions(train_prices, y_true_unscaled,
                                  predicted_unscaled)
    mlflow.log_figure(fig_dist, "distributions.png")

    # calculate metrics
    reg_metrics = calculate_metrics(y_true_unscaled, predicted_unscaled)
    mlflow.log_metrics(reg_metrics)

    # return log mae
    return reg_metrics["log_mae"], reg_metrics["r2"]


if __name__ == "__main__":
    study = optuna.create_study(study_name="catboost_hyperparam",
                                directions=["minimize", "maximize"])
    study.optimize(objective,
                   n_trials=50,
                   timeout=None,
                   callbacks=[mlflc],
                   n_jobs=1)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")

    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    os.system("shutdown.exe -s -t 0")
