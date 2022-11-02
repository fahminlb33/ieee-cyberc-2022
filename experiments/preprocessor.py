import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

NUM_ATTRS = ["#Floors", "#Rooms", "#Halls", "Area"]
CAT_ATTRS = ["City", "District", "Street", "Community", "Floor", "Orient"]


class DataPreprocessor:

    def __init__(self,
                 timeframe=["2017-01-01", "2018-12-31"],
                 drop_columns=["Orient", "Lat", "Lon"],
                 drop_nulls=True,
                 scale=True,
                 remove_outliers=None,
                 moving_average=None,
                 moving_average_groupby=None):
        self.timeframe = timeframe
        self.drop_columns = drop_columns
        self.drop_nulls = drop_nulls
        self.scale = scale
        self.remove_outliers = remove_outliers
        self.moving_average = moving_average
        self.moving_average_groupby = moving_average_groupby

    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        # copy dataframes
        train_pool = df_train.copy()
        test_pool = df_test.copy()

        # construct real price, make sure it is not at the end of our dataframe
        # train_pool.insert(train_pool.shape[1] - 1, "RealPrice", train_pool["Price"] * train_pool["Area"])
        # test_pool.insert(test_pool.shape[1] - 1, "RealPrice", test_pool["Price"] * test_pool["Area"])

        train_pool["Price"] = train_pool["Price"] * train_pool["Area"]
        test_pool["Price"] = test_pool["Price"] * test_pool["Area"]

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

        # make sure to drop Time column
        train_pool = train_pool.drop(columns=["Time"])
        test_pool = test_pool.drop(columns=["Time"])

        # transform into numpy arrays
        X_train = train_pool.iloc[:, :-1].values
        y_train = train_pool.iloc[:, -1].values
        X_test = test_pool.iloc[:, :-1].values
        y_test = test_pool.iloc[:, -1].values

        # get categorical column indexes
        cat_columns = [x for x in CAT_ATTRS if x not in self.drop_columns]
        cat_cols_idx = [
            i for i, col in enumerate(train_pool.columns) if col in cat_columns
        ]

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.categorical_column_indexes = cat_cols_idx
        self.columns = train_pool.columns.tolist()

    def inverse_transform_price(self, y_pred):
        if self.scale:
            return self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        else:
            return y_pred
