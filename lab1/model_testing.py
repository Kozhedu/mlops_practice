from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
lin_reg = pickle.load(open('phish.pkl', 'rb'))
x_test_lin = pd.read_csv("X_test.csv")
y_test_lin = pd.read_csv("y_test.csv")
y_predict_lin=lin_reg.predict(x_test_lin)
mse = mean_squared_error(y_test_lin, y_predict_lin)
print("Mean squared error: %.2f" % mse)
