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

import argparse

import pandas as pd
from catboost import CatBoostRegressor, Pool

from experiments.preprocessor import DataPreprocessorV2

if __name__ == "__main__":
    # create CLI parser
    cli_parser = argparse.ArgumentParser(description='CatBoost Experiments')
    cli_parser.add_argument(
        'model_path',
        type=str,
        help=
        'Path to CatBoost model (CBM).'
    )
    cli_parser.add_argument('input_path',
                            type=str,
                            help='Input data to predict without the Price column as CSV file.')
    cli_parser.add_argument('output_path',
                            type=str,
                            help='Prediction output path to CSV file.')

    # parse CLI arguments
    args = cli_parser.parse_args()

    # --- step 1 - load and preprocess dataset
    print(f"Input dataset: {args.input_path}")
    df = pd.read_csv(args.input_path, parse_dates=["Time"])
    
    # preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessorV2(
        timeframe=[],
        drop_columns=["Id", "Time", "Lat", "Lon", "Orient"],
        drop_nulls=False,
        scale=False)

    # transform data
    # no need to fit since we only required to drop some columns
    df_input = preprocessor.transform(df, train=False)

    # create dataset pool
    test_pool = Pool(df_input["X"],
                        df_input["y"],
                        cat_features=df_input["cat_names_index"], timestamp=df_input["timestamp"])

    # --- step 2 - load model and run prediction
    print(f"Loading model: {args.model_path}")
    model = CatBoostRegressor()
    model.load_model(args.model_path)
    
    # run predictions
    print("Running predictions...")
    predicted = model.predict(test_pool)
    predicted_unscaled = preprocessor.unscale_price(predicted.reshape(-1, 1))

    # save predictions
    print("Writing predictions...")
    df_pred = pd.DataFrame({
        "id": df["Id"],
        "prediction": predicted_unscaled.flatten()
    })
    df_pred.to_csv(args.output_path, index=False)

    print("Prediction completed!")
