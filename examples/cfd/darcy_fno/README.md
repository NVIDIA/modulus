## train an FNO model for the Dary problem
To train the model, run
```bash
python train_fno_darcy.py
```
training data will be generated on the fly.

Progress can be monitored using MLFlow. Open a new terminal
and navigate to the training directory, then run
```bash
mlflow ui -p 2458
```
View progress in a browser at http://127.0.0.1:2458
