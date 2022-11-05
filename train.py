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

import os
import json
import argparse

import pandas as pd
from catboost import CatBoostRegressor, Pool

from experiments.preprocessor import DataPreprocessorV2
from experiments.helpers import calculate_metrics, print_metrics, plot_all


def find_dataset_paths(dataset_path: str) -> tuple[str, str]:
    files = os.listdir(dataset_path)
    train_filename = next((x for x in files if 'train' in x), None)
    test_filename = next((x for x in files if 'test' in x), None)

    train_path = os.path.join(dataset_path, train_filename)
    test_path = os.path.join(dataset_path, test_filename)

    return train_path, test_path

if __name__ == "__main__":
    # create CLI parser
    cli_parser = argparse.ArgumentParser(description='CatBoost Experiments')
    cli_parser.add_argument(
        'dataset',
        type=str,
        help=
        'Training and test dataset path. Must contains the word "train" and "test" in the filename.'
    )
    cli_parser.add_argument('output_path',
                            type=str,
                            help='Output model and prediction path.')

    # parse CLI arguments
    args = cli_parser.parse_args()

    # run all
    os.makedirs(args.output_path, exist_ok=True)

    # --- step 1 - find dataset path
    train_path, test_path = find_dataset_paths(args.dataset)
    if train_path is None:
        print("No training dataset found.")
        exit(1)
    if test_path is None:
        print("No test dataset found.")
        exit(1)

    print(f"Training dataset: {train_path}")
    print(f"Test dataset: {test_path}")

    # --- step 2 - load and preprocess data
    print("Loading data...")
    df_train = pd.read_csv(train_path, parse_dates=["Time"])
    df_test = pd.read_csv(test_path, parse_dates=["Time"])
    
    # preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessorV2(
        timeframe=[],
        drop_columns=["Id", "Time", "Lat", "Lon", "Orient"],
        drop_nulls=True,
        scale=False)
    preprocessor.fit(df_train)

    # transform data
    df_train_p = preprocessor.transform(df_train, train=True)
    df_test_p = preprocessor.transform(df_test, train=False)

    # create dataset pool
    train_pool = Pool(df_train_p["X"],
                        df_train_p["y"],
                        cat_features=df_train_p["cat_names_index"], timestamp=df_train_p["timestamp"])
    test_pool = Pool(df_test_p["X"],
                        df_test_p["y"],
                        cat_features=df_test_p["cat_names_index"], timestamp=df_test_p["timestamp"])

    # --- step 3 - run training
    # create profile
    train_params = {
        "has_time": True,
        "random_seed": 42,
        "task_type": "GPU",
        "iterations": 1000,
        "loss_function": "RMSE",
        "depth": 15,
        "l2_leaf_reg": 1.5,
        "border_count": 128,
        "grow_policy": "SymmetricTree",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.939114487427496,
    }

    # Create model
    print("Training model...")
    model = CatBoostRegressor(**train_params)
    model.fit(train_pool)

    # save moel
    model.save_model(os.path.join(args.output_path, "model.cbm"))

    # --- step 4 - evaluate model
    # evaluate model
    print("Evaluating model...")
    predicted = model.predict(test_pool)
    predicted_unscaled = preprocessor.unscale_price(predicted.reshape(-1, 1))
    y_true_unscaled = df_test["Price"].values.reshape(-1, 1)
    
    # calculate metrics

    # print metrics and params
    print("Metrics:")
    print_metrics(y_true_unscaled, predicted_unscaled)
    
    out_path = os.path.join(args.output_path, "metrics.json")
    with open(out_path, "w") as f:
        metrics = calculate_metrics(y_true_unscaled, predicted_unscaled)
        json.dump(metrics, f)
    
    out_path = os.path.join(args.output_path, "model_params.json")
    with open(out_path, "w") as f:
        json.dump(model.get_all_params(), f)
    
    out_path = os.path.join(args.output_path, "feature_importances.json")
    with open(out_path, "w") as f:
        feature_importances = dict(zip(df_train_p["columns"], model.feature_importances_))
        json.dump(feature_importances, f)

    # save predictions
    print("Writing predictions...")
    df_pred = pd.DataFrame({
        "id": df_test["Id"],
        "y_true": df_test["Price"],
        "y_pred": predicted_unscaled.flatten()
    })
    df_pred.to_csv(os.path.join(args.output_path, "predictions.csv"), index=False)

    # plot scatterplot
    fig = plot_all(y_true_unscaled, predicted_unscaled, max_ticks=6)
    fig.savefig(os.path.join(args.output_path, "plots.png"))

    print("Training and evaluation completed!")
