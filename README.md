# LR_PSO_Tensorflow

This project try to use the simple PSO to solve the logistic regression (LR) model in Tensorflow. Compring to the original LR scripts, this project use the Iris dataset insteaded of the MNIST to help the user to understand more about how Tensorflow access the data.

The scripts are run in envirorment :
1. python3
2. tensorflow 1.3

To run the script, just type
```python
python3 logistic_regression.py # this is original version
python3 logistic_regression_simplepso.py # this version use PSO
python3 logistic_regression_simplepso_batch.py # you may also want to use mini-batch training
```