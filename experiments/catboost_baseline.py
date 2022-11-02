import json
import argparse

import mlflow

import pandas as pd
from catboost import CatBoostRegressor, Pool

from helpers import calculate_metrics, plot_scatterplot, plot_distributions
from preprocessor import DataPreprocessor

if __name__ == "__main__":
    # get run name
    arg_parser = argparse.ArgumentParser(description='CatBoost Experiments')
    arg_parser.add_argument('name', type=str, help='Run name')
    args = vars(arg_parser.parse_args())

    # setup mlflow experiment
    mlflow.set_experiment("catboost2")

    # load data
    print("Loading data...")
    df_train = pd.read_csv('./dataset/house_price_train.csv',
                           parse_dates=["Time"])
    df_test = pd.read_csv('./dataset/house_price_test.csv',
                          parse_dates=["Time"])

    # preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(
        timeframe=['2017-01-01', '2018-12-31'],
        drop_columns=["Id", "Lat", "Lon", "Orient"],
        drop_nulls=True,
        scale=True,
        remove_outliers=None,
        moving_average=None,
        moving_average_groupby=None)
    preprocessor.fit(df_train, df_test)

    # start mlflow experiment
    with mlflow.start_run(run_name=args['name']):
        # log parameters
        mlflow.log_param("timeframe", preprocessor.timeframe)
        mlflow.log_param("drop_columns", preprocessor.drop_columns)
        mlflow.log_param("drop_nulls", preprocessor.drop_nulls)
        mlflow.log_param("scale", preprocessor.scale)
        mlflow.log_param("remove_outliers", preprocessor.remove_outliers)
        mlflow.log_param("moving_average", preprocessor.moving_average)
        mlflow.log_param("moving_average_groupby",
                         preprocessor.moving_average_groupby)

        # create dataset pool
        train_pool = Pool(preprocessor.X_train,
                          preprocessor.y_train,
                          cat_features=preprocessor.categorical_column_indexes)
        test_pool = Pool(preprocessor.X_test,
                         preprocessor.y_test,
                         cat_features=preprocessor.categorical_column_indexes)

        # train parameters
        train_params = {
            "random_seed": 42,
            "task_type": "GPU",
            "loss_function":
            "RMSE",  # RMSE, Tweedie:variance_power=1.9, Poisson
            "one_hot_max_size": None,  # None, 255
            "iterations": 1000,  # 1000
            "learning_rate": None,  # None, 0.03, 0.1
            "depth": None,  # 6, 8
            "l2_leaf_reg": None,  # 3.0
            "bootstrap_type": "Bayesian",  # Bayesian
        }
        mlflow.log_params(train_params)

        # Create model
        print("Training model...")
        model = CatBoostRegressor(**train_params)
        model.fit(train_pool)

        # save model artifact
        feature_importances = {
            name: value
            for name, value in zip(preprocessor.columns,
                                   model.feature_importances_)
        }
        mlflow.log_text(json.dumps(feature_importances),
                        "feature_importances.json")
        mlflow.log_text(json.dumps(model.get_all_params()),
                        "model_params.json")
        mlflow.catboost.log_model(model, "model")

        # evaluate model
        print("Evaluating model...")
        predicted = model.predict(test_pool)
        predicted_unscaled = preprocessor.inverse_transform_price(
            predicted.reshape(-1, 1))
        y_true_unscaled = preprocessor.inverse_transform_price(
            preprocessor.y_test.reshape(-1, 1))

        # save metrics
        mlflow.log_metrics(
            calculate_metrics(y_true_unscaled, predicted_unscaled))

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
