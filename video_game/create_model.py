import pandas as pd
from keras.models import Sequential
from keras.layers import *
from sklearn.externals import joblib

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# create the model
model = Sequential()
model.add(Dense(50,input_dim=9,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(1,activation = 'linear'))
model.compile('adam','mean_squared_error')

# train the model
model.fit(
    X,
    Y,
    epochs = 50,
    shuffle = True,
    verbose = 2
)

# Load the separate test data set
test_data_df = pd.read_csv("sales_data_test_scaled.csv")
X_test = test_data_df.drop('total_earnings',axis = 1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose = 0)
print("The mean squared error (MSE) for the test data set is: {} ".format(test_error_rate))

# Make a prediction with the neural network
X = pd.read_csv("../04/proposed_new_product.csv").values
prediction = model.predict(X)
prediction = prediction[0][0]

# Rescale the prediction result
print(prediction)
scaler = joblib.load("min_max_scaler")
prediction = prediction + scaler.min_[8]
prediction = prediction / scaler.scale_[8]

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
