from sklearn.datasets import make_classification
import pandas as pd
X, y = make_classification(
    n_samples=366, 
    n_features=5, 
    n_informative=3, 
    n_classes=2, 
    random_state=999 )
date = pd.date_range(start='2022-01-01', end='2023-01-01')
train = pd.DataFrame(columns=['Date', 'Tem'])
train['Date'] = pd.DataFrame(date)
test = pd.DataFrame(columns=['Date', 'Tem'])
test['Date'] = pd.DataFrame(date)
test['Tem'] = pd.DataFrame(X[: , 2])
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)