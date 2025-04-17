import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
from datetime import datetime
import numpy as np
from metrics import save_classification_report, save_conf_mat
from process import prepare_datasets, get_predictions
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv("DATABASE_PATH")
time = datetime.now().strftime("%y-%m-%dT%H-%M")
model_name = "resnet"

# Set the number of training samples
num_train = 1000000

# Prepare datasets
X_train, y_train, X_test, y_test, X_val, y_val, label_encoder = prepare_datasets(data_path, num_train)

# Build the model by stacking the base model and our custom layers on top.
model = Sequential([
    ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
optim = Adam()
model.compile(optimizer=optim,
              loss="binary_crossentropy",
              metrics=['accuracy'])

# Set early stopping to reduce overfitting
early_stop = EarlyStopping(monitor='val_loss',
                           patience=3,
                           restore_best_weights=True)

# Fit on training data
model.fit(X_train, y_train, 
          epochs=10, 
          validation_data=(X_test, y_test),
          callbacks=[early_stop])

# Predict on the test and validation sets
y_test_pred = model.predict(X_test)
y_val_pred = model.predict(X_val)

# Get string values back from predictions
test_pred, test_true, val_pred, val_true = get_predictions(y_test_pred, y_test, y_val_pred, y_val, label_encoder)

# Save reports and confusion matrices
save_classification_report(test_true, test_pred,  model_name + "_test" , time)
save_conf_mat(test_true, test_pred, model_name + "_test", time)

save_classification_report(val_true, val_pred, model_name + "_val", time)
save_conf_mat(val_true, val_pred, model_name + "_val", time)