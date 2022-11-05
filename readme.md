# IEEE CyberC Data Analytics Competition 2022

Author: Fahmi Noor Fiqri, Student Branch Pakuan University

This is my attempt in participating in IEEE CyberC 2022 Data Analytics Competition.

The instructions on how to run the modeling process and making predictions is explained on this README. To get the pretrained model along with the metrics and parameters, please refer to my submitted report.

## Running The Program

Clone this repository first! Then before running the training process and making predictions, you need to create a new virtual environment with all the packages listed in the `requirements.txt`. I recommend using Anaconda to create the virtual environment.

```sh
pip install -r requirements.txt
```

Note: The `requirements.txt` only contains the packages to run the `train.py` and `predict.py`. If you want to run the experiments in the `notebooks` directory, you must install these additional packages: `notebook ipykernel tensorflow`.

### Training New Model

`cd` to the root folder of this repository, then run `python train.py <DATASET_PATH>  <OUTPUT_PATH>`. The dataset path MUST contains the `train` and `test` data.

```sh
$ python train.py ./dataset ./final_model

Training dataset: dataset/house_price_train.csv
Test dataset: dataset/house_price_test.csv
Loading data...
Preprocessing data...
Warning: drop_nulls and fill_nulls are ignored when train=False
Training model...
0:      learn: 19313.2053547    total: 233ms    remaining: 3m 52s
1:      learn: 18898.6887122    total: 473ms    remaining: 3m 56s
...
998:    learn: 4212.8398553     total: 3m 56s   remaining: 237ms
999:    learn: 4210.0306267     total: 3m 56s   remaining: 0us
Evaluating model...
Metrics:
Min:  4999.2346
Max:  129765.2845
R2:  0.8928
MAPE:  0.1393
Log MAE:  0.0
Writing predictions...
Training and evaluation completed!
```

The finalized model, metrics, and other parameters are stored in the output path. The model parameters is also adjusted to the best parameters from hyperparameter search using `optuna`.

### Making Prediction

To make predictions from input file, run `python predict.py <MODEL_PATH> <INPUT_PATH> <OUTPUT_PATH>`

- `<MODEL_PATH>` should be an absolute path to the model file (CBM)
- `<INPUT_PATH>`  should be an absolute path to the input CSV file and it MUST have the same column names and order as in the original dataset
- `OUTPUT_PATH` should be an absolute path to the predictions output as CSV file. The output CSV will include two columns, `id` and `prediction`

```sh
$ python predict.py ./final_model/model.cbm ./dataset/house_price_test.csv ./predictions.csv
Input dataset: ./dataset/house_price_test.csv
Preprocessing data...
Loading model: ./final_model/model.cbm
Running predictions...
Writing predictions...
Prediction completed!
```

Sample predictions CSV:

```csv
id,prediction
553098,47169.76688359724
553099,46207.49780653464
553100,45355.528640414705
```

## License

This project is licensed under Apache License 2.0.
