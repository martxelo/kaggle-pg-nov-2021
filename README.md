# Tabular Playground Series - Nov 2021

This repo allows to fit a neural network model (multi layer perceptron) with [TensorFlow](https://www.tensorflow.org/) for the [Kaggle tabular competition of november 2021](https://www.kaggle.com/c/tabular-playground-series-nov-2021/). It uses [MLfow](https://mlflow.org/) for tracking experiments and runs and keeps a record of parameters and metrics for every run.

## Download

Download the repo with:
```bash
user@laptop:~$ git clone https://github.com/martxelo/kaggle-pg-nov-2021.git
```

## Configure

In the project folder create a python virtual environment and activate it:

```bash
user@laptop:~$ cd kaggle-pg-nov-2021
user@laptop:~/kaggle-pg-nov-2021$ python -m venv .env --prompt kgl-env
user@laptop:~/kaggle-pg-nov-2021$ source .env/bin/activate
(kgl-env) user@laptop:~/kaggle-pg-nov-2021$
```

For windows users replace the activation command with:
```console
C:\Users\user\kaggle-pg-nov-2021>.env\Scripts\activate.bat
```

Install all dependencies:

```bash
(kgl-env) user@laptop:~/kaggle-pg-nov-2021$ pip install -r requirements.txt
```

## Usage

To see the help run this from the project directory:

```
(kgl-env) user@laptop:~/kaggle-pg-nov-2021$ python models/tf_nn.py -h

Train a neural network classifier

optional arguments:
  -h, --help            show this help message and exit
  --nrows int           Fraction of data for testing models
  --test_size float     Select fraction of test data
  --n_components float  Select number of components for PCA
  --hidden_layer_sizes tuple [tuple ...]
                        Neurons in hidden layers
  --activation string   Activation function for hidden layers
  --stddev float        Gaussian noise for not overfitting
  --batch_size int      Batch_size for fitting the model
```

---
**NOTE**

If your computer does not have GPU a warning is thrown (ignore the message):

```
W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory

I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
```
---

All the parameters have default values. You can use other parameters for training a model. For example, to train a model with three hidden layers with (60, 50, 10) neurons and 'tanh' activation function run:

```
(kgl-env) user@laptop:~/kaggle-pg-nov-2021$ python models/tf_nn.py --hidden_layer_sizes 60 50 10 --activation tanh
```

Train another model with:

```
(kgl-env) user@laptop:~/kaggle-pg-nov-2021$ python models/tf_nn.py --hidden_layer_sizes 40 30 20 --activation relu
```

## Tracking models

Now you can see the model with the MLflow user interface. Run:

```
(kgl-env) user@laptop:~/kaggle-pg-nov-2021$ mlflow ui
```

And go to your web browser http://127.0.0.1/5000. You will see this:

![Main page of MLflow](/images/main.png)

You can sort runs by a metric or a parameter, filter the runs, etc. Click on a run to see the info for the run:

![Run 01](/images/run01.png)

The project keeps a record of the AUC plot and the sample_submission.csv file. You can download the submission file and upload it to Kaggle.

## Parameters

Additional info for tunning the parameters:

- `nrows`: the number of rows used when reading the train.csv file. It is a random sample extracted from the DataFrame. To use all the data `nrows=-1`. Default 10000.
- `test_size`: the fraction of the data use as validation set. Default 0.2.
- `n_components`: if integer higher than 1 then it is the number of components in the PCA transformation. If float between 0 and 1 then it is the explained variance to keep after PCA transformation. More info [here](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Default 0.95.
- `hidden_layer_sizes`: the number of neurons in the hidden layers. You can change the number of neurons and the number of layers. Default [50, 40, 3].
- `activation`: the activation function for the hidden layers (the last layer has always a 'sigmoid' activation function). It is possible to use any of [these](https://www.tensorflow.org/api_docs/python/tf/keras/activations). Default 'relu'.
- `stddev`: standard deviation for the first layer. There is a [GaussianNoise](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GaussianNoise) layer to avoid overfitting during training. Default 0.01.
- `batch_size`: batch size during training. Default 1024.

It is easy to add more parameters and track them with MLflow. Different scaler, other steps like [polynomial features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html), etc.

## Try many models

There is a script to run many models (run_models.py). Change the values for the parameters or add more and run all the possible combinations. This may take several hours.

## Other models

There are two examples of other models in the models folder: scikit-learn DecissionTreeClassifier and scikit-learn MLPClassifier. If you try these models they are going to be tracked in another experiment with their own parameters and metrics. Other libraries like [PyTorch](https://pytorch.org/), [LightGBM](https://lightgbm.readthedocs.io/en/latest/), [XGBoost](https://xgboost.readthedocs.io/en/stable/) can be used following the same structure.