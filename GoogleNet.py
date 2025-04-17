import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from tensorflow.keras.optimizers import Adam  
from metrics import save_classification_report, save_conf_mat
from process import prepare_datasets, get_predictions
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv("DATABASE_PATH")
time = datetime.now().strftime("%y-%m-%dT%H-%M")
model_name = "googleNet"

num_train = 100000

# Image size (299,299) to match InceptionV3 input requirements
X_train, y_train, X_test, y_test, X_val, y_val, encoder = prepare_datasets(data_path, num_train, (299,299))

# Build the model by stacking the base model and our custom layers on top.
model = Sequential([
    InceptionV3(include_top=False, weights="imagenet", input_shape=(299, 299, 3)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
optim = Adam()
model.compile(optimizer=optim,
              loss="binary_crossentropy",
              metrics=['accuracy'])

# Add early stopping to prevent overfitting
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

# Convert predictions to categorical labls
test_pred, test_true, val_pred, val_true = get_predictions(y_test_pred, y_test, y_val_pred, y_val, encoder)

# Save reports and confusion matrices
save_classification_report(test_true, test_pred,  model_name + "_test" , time)
save_conf_mat(test_true, test_pred, model_name + "_test", time)

save_classification_report(val_true, val_pred, model_name + "_val", time)
save_conf_mat(val_true, val_pred, model_name + "_val", time)