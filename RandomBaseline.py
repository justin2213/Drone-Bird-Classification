import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from datetime import datetime
from metrics import save_classification_report, save_conf_mat
from process import prepare_datasets, get_predictions
from dotenv import load_dotenv
from process import load_images
import random

load_dotenv()

data_path = os.getenv("DATABASE_PATH")
time = datetime.now().strftime("%y-%m-%dT%H-%M")
model_name = "baseline"

# Number of train, test, and val samples
num_train = 100000
num_test = int(num_train * 0.15)
num_val = int(num_train * 0.15)

# Load data 
print("Loading testing data")
_, y_test = load_images(dir=os.path.join(data_path, "test"), limit=num_test)
print("Loading validation data")
_, y_val = load_images(dir=os.path.join(data_path, "valid"), limit=num_val)

# Iterate, predicting a random guess each time
y_test_pred = []
for i in range(len(y_test)):
  guess = random.random()
  if guess >= 0.5:
    y_test_pred.append('Bird')
  else:
    y_test_pred.append('Drone')

# Do it again for validation set
y_val_pred = []
for i in range(len(y_val)):
  guess = random.random()
  if guess >= 0.5:
    y_val_pred.append('Bird')
  else:
    y_val_pred.append('Drone')

# Save the classification report and confusion matrix
save_classification_report(y_test, y_test_pred, "test_" + model_name, time)
save_conf_mat(y_test, y_test_pred, "test_" + model_name, time)

save_classification_report(y_val, y_val_pred, "val_" + model_name, time)
save_conf_mat(y_val, y_val_pred, "val_" + model_name, time)

