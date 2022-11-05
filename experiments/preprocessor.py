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

from typing import Union

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

NUM_ATTRS = ["#Floors", "#Rooms", "#Halls", "Area"]
CAT_ATTRS = ["City", "District", "Street", "Community", "Floor", "Orient"]


class DataPreprocessorV2:

    def __init__(self,
                 timeframe=["2017-01-01", "2018-12-31"],
                 scale=True,
                 drop_columns=["Orient", "Lat", "Lon"],
                 drop_nulls=True,
                 fill_nulls=False,
                 remove_outliers=None):
        self.timeframe = timeframe
        self.scale = scale
        self.drop_columns = drop_columns
        self.drop_nulls = drop_nulls
        self.fill_nulls = fill_nulls
        self.remove_outliers = remove_outliers

    def fit(self, df: pd.DataFrame):
        # outlier detection
        prices = df["Price"].values.reshape(-1, 1)
        if self.remove_outliers == "isolation_forest":
            self.isolation_forest_clf = IsolationForest(
                n_estimators=100, warm_start=True, random_state=0).fit(prices)
        elif self.remove_outliers == "z_score":
            self.z_scores = (prices - prices.mean()) / prices.std()
        elif self.remove_outliers == "iqr":
            q1, q3 = np.percentile(prices, [25, 75])
            self.q1 = q1
            self.q3 = q3

        # standardize target feature
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(prices)

    def transform(self,
                  df: pd.DataFrame,
                  train=True) -> dict[str, Union[np.ndarray, list]]:
        # copy DataFrame
        df_pool = df.copy()

        # set Id as index
        df_pool = df_pool.set_index("Id")

        # copy timestamps
        timestamps = df_pool["Time"].copy()

        # if train=False, then we don't need to select by date, we want to process it all
        if train:
            # select by date
            if len(self.timeframe) == 2:
                df_pool = df_pool.loc[self.timeframe[0]:self.timeframe[1]]

            # outlier detection
            # ---- Isolation Forest ----
            prices = df["Price"].values.reshape(-1, 1)
            if self.remove_outliers == "isolation_forest":
                inliers = self.isolation_forest_clf.predict(prices)

                inliers_index = df["Id"].values[inliers == 1]
                df_pool = df_pool[df_pool["Id"].isin(inliers_index)]

            # ---- Z-Score ----
            elif self.remove_outliers == "z_score":
                inliers_index = df_pool["Id"].values[np.ravel(
                    np.abs(self.z_scores) < 3)]
                df_pool = df_pool[df_pool["Id"].isin(inliers_index)]

            # ---- IQR ----
            elif self.remove_outliers == "iqr":
                iqr = self.q3 - self.q1
                iqr_indexes = (prices > self.q1 - 1.5 * iqr) & (
                    prices < self.q3 + 1.5 * iqr)

                inliers_index = df_pool["Id"].values[np.ravel(iqr_indexes)]
                df_pool = df_pool[df_pool["Id"].isin(inliers_index)]
            
            # drop unused columns
            if len(self.drop_columns) > 0:
                df_pool = df_pool.drop(columns=self.drop_columns, errors="ignore")

            # drop nulls
            if self.drop_nulls:
                df_pool = df_pool.dropna()

            # fill nulls
            if self.fill_nulls:
                df_pool = df_pool.fillna("")
        else:
            # drop unused columns
            if len(self.drop_columns) > 0:
                df_pool = df_pool.drop(columns=self.drop_columns, errors="ignore")

            # fill nulls regardless
            if self.drop_nulls or self.fill_nulls:
                print("Warning: drop_nulls and fill_nulls are ignored when train=False")
                df_pool = df_pool.fillna("")

        # filter timestamp by index
        timestamps = timestamps.loc[df_pool.index].values.reshape(-1, 1)

        # split into X and y
        X = df_pool.iloc[:, :-1].values
        y = df_pool.iloc[:, -1].values.reshape(-1, 1)

        # scale target feature
        if self.scale:
            y = self.scaler.transform(y)

        # get categorical features index
        cat_features_index = [
            i for i, col in enumerate(df_pool.columns) if col in CAT_ATTRS
        ]

        return {
            "dataframe": df_pool,
            "X": X,
            "y": y,
            "columns": df_pool.columns.tolist(),
            "cat_names": df_pool.columns[cat_features_index].tolist(),
            "cat_names_index": cat_features_index,
            "index": df_pool.index.values,
            "timestamp": timestamps
        }

    def unscale_price(self, y_pred) -> np.ndarray:
        if self.scale:
            return self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        else:
            return y_pred


class DataPreprocessor:

    def __init__(self,
                 timeframe=["2017-01-01", "2018-12-31"],
                 drop_columns=["Orient", "Lat", "Lon"],
                 scale=True,
                 drop_nulls=True,
                 fill_nulls=False,
                 remove_outliers=None,
                 moving_average=None,
                 moving_average_groupby=None):
        self.timeframe = timeframe
        self.drop_columns = drop_columns
        self.scale = scale
        self.drop_nulls = drop_nulls
        self.fill_nulls = fill_nulls
        self.remove_outliers = remove_outliers
        self.moving_average = moving_average
        self.moving_average_groupby = moving_average_groupby

    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        # copy dataframes
        train_pool = df_train.copy()
        test_pool = df_test.copy()

        # select by date
        if len(self.timeframe) == 2:
            train_pool = train_pool.loc[self.timeframe[0]:self.timeframe[1]]

        # remove outliers
        if self.remove_outliers == "isolation_forest":
            indexes = train_pool["Id"].values
            prices = train_pool["Price"].values.reshape(-1, 1)

            isolation_forest_clf = IsolationForest(n_estimators=100,
                                                   warm_start=True,
                                                   random_state=0).fit(prices)
            inliers = isolation_forest_clf.predict(prices)

            inliers_index = indexes[inliers == 1]
            train_pool = train_pool[train_pool["Id"].isin(inliers_index)]
        elif self.remove_outliers == "z_score":
            indexes = train_pool["Id"].values
            prices = train_pool["Price"].values.reshape(-1, 1)

            z_scores = (prices - prices.mean()) / prices.std()

            inliers_index = indexes[np.ravel(np.abs(z_scores) < 3)]
            train_pool = train_pool[train_pool["Id"].isin(inliers_index)]
        elif self.remove_outliers == "iqr":
            indexes = train_pool["Id"].values
            prices = train_pool["Price"].values.reshape(-1, 1)

            q1, q3 = np.percentile(prices, [25, 75])
            iqr = q3 - q1

            iqr_indexes = (prices > q1 - 1.5 * iqr) & (prices < q3 + 1.5 * iqr)
            inliers_index = indexes[np.ravel(iqr_indexes)]
            train_pool = train_pool[train_pool["Id"].isin(inliers_index)]

        # drop unused columns
        if len(self.drop_columns) > 0:
            train_pool = train_pool.drop(columns=self.drop_columns)
            test_pool = test_pool.drop(columns=self.drop_columns)

        # drop nulls
        if self.drop_nulls:
            train_pool = train_pool.dropna()
            test_pool = test_pool.dropna()

        # fill nulls
        if self.fill_nulls:
            train_pool = train_pool.fillna("NA")
            test_pool = test_pool.fillna("NA")

        # construct moving averages
        if self.moving_average is not None:
            train_pool = train_pool.set_index("Time")
            if self.moving_average_groupby is not None:
                train_pool["Price"] = train_pool.groupby(
                    self.moving_average_groupby)["Price"].transform(
                        lambda x: x.rolling(self.moving_average).mean())
            else:
                train_pool["Price"] = train_pool.rolling(
                    self.moving_average)["Price"].mean()

        # standardize target feature
        scaler = None
        if self.scale:
            scaler = StandardScaler()
            # scaler = LogScaler()
            train_pool["Price"] = scaler.fit_transform(
                train_pool["Price"].values.reshape(-1, 1))
            test_pool["Price"] = scaler.transform(
                test_pool["Price"].values.reshape(-1, 1))

        # save Ids
        self.test_ids = test_pool["Id"].values

        # make sure to drop Time column
        train_pool = train_pool.drop(columns=["Id", "Time"])
        test_pool = test_pool.drop(columns=["Id", "Time"])

        # transform into numpy arrays
        X_train = train_pool.iloc[:, :-1].values
        y_train = train_pool.iloc[:, -1].values
        X_test = test_pool.iloc[:, :-1].values
        y_test = test_pool.iloc[:, -1].values

        # get categorical column indexes
        cat_cols_idx = [
            i for i, col in enumerate(train_pool.columns) if col in CAT_ATTRS
        ]

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.categorical_column_indexes = cat_cols_idx
        self.columns = train_pool.columns.tolist()

    def inverse_transform_price(self, y_pred) -> np.ndarray:
        if self.scale:
            return self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        else:
            return y_pred
