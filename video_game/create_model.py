import pandas as pd
from keras.models import Sequential
from keras.layers import *
from sklearn.externals import joblib
import keras
import tensorflow as tf

RUN_NAME = "run 2 with 5 nodes"

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# create the model
model = Sequential()
model.add(Dense(5,input_dim=9,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(1,activation = 'linear'))
model.compile('adam','mean_squared_error')

# create logger

logger = keras.callbacks.TensorBoard(
    log_dir="logs/{}".format(RUN_NAME),
    write_grads= True,
    histogram_freq= 5
)

# train the model
model.fit(
    X,
    Y,
    epochs = 50,
    shuffle = True,
    verbose = 2,
    callbacks = [logger]
)

# Load the separate test data set
test_data_df = pd.read_csv("sales_data_test_scaled.csv")
X_test = test_data_df.drop('total_earnings',axis = 1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose = 0)
print("The mean squared error (MSE) for the test data set is: {} ".format(test_error_rate))

model.save("trained_model.h5")

# Export model in google cloud format
model_builder = tf.saved_model.builder.SavedModelBuilder("exported_model")

inputs = {
    'input':tf.saved_model.utils.build_tensor_info(model.input)
}

outputs = {
    'earnings': tf.saved_model.utils.build_tensor_info(model.output)
}

signature_def = tf.saved_model.signature_def_utils.build_signature_def(
    inputs = inputs,
    outputs = outputs,
    method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)

model_builder.add_meta_graph_and_variables(
    K.get_session(),
    tags = [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }
)

model_builder.save()

# Make a prediction with the neural network
X = pd.read_csv("proposed_new_product.csv").values
prediction = model.predict(X)
prediction = prediction[0][0]

# Rescale the prediction result
print(prediction)
scaler = joblib.load("min_max_scaler")
prediction = prediction + scaler.min_[8]
prediction = prediction / scaler.scale_[8]

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
