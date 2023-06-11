from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
import numpy as np
predict_good = ['youtube.com']
loaded_model = pickle.load(open('phish.pkl', 'rb'))
result = loaded_model.predict(predict_good)
print(result)
