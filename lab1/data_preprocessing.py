from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(train.drop('Date', axis = 1), test.drop('Date', axis = 1), test_size = 0.33, random_state = 42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)