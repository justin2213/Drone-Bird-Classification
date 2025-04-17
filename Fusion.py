import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
from process import (
    load_images_and_extract_features,
    encode_labels_train,
    encode_labels_transform,
    undo_label_encoding,
)
from metrics import save_classification_report, save_conf_mat

load_dotenv()

data_path = os.getenv("DATABASE_PATH")
time = datetime.now().strftime("%y-%m-%dT%H-%M")
model_name = "fusion"

num_train = 1000000
num_test = int(num_train * 0.15)
num_val = int(num_train * 0.15)

# Load features
X_train, y_train_raw = load_images_and_extract_features(os.path.join(data_path, "train"), num_train)
X_test, y_test_raw   = load_images_and_extract_features(os.path.join(data_path, "test"),  num_test)
X_val, y_val_raw     = load_images_and_extract_features(os.path.join(data_path, "valid"), num_val)

# Encode the labels
y_train, label_encoder = encode_labels_train(y_train_raw)
y_test = encode_labels_transform(y_test_raw, label_encoder)
y_val  = encode_labels_transform(y_val_raw, label_encoder)

# Define ml pipeline
pipeline = make_pipeline(
    StandardScaler(),
    XGBClassifier(
        random_state=42,
        colsample_bytree=0.9,
        max_depth=3,
        learning_rate=0.2,
        n_estimators=300,
        subsample=0.8,
        min_child_weight=3,
        reg_alpha=0.01,
    ),
    verbose=True
)

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict on test and validation sets 
y_test_pred_num = pipeline.predict(X_test)
y_val_pred_num  = pipeline.predict(X_val)

# Undo label encoding
test_pred = undo_label_encoding(y_test_pred_num, label_encoder)
test_true = undo_label_encoding(y_test,          label_encoder)

val_pred = undo_label_encoding(y_val_pred_num, label_encoder)
val_true = undo_label_encoding(y_val,         label_encoder)

# Save classification report and confusion matrix
save_classification_report(test_true, test_pred,  model_name + "_test" , time)
save_conf_mat(test_true, test_pred, model_name + "_test", time)

save_classification_report(val_true, val_pred, model_name + "_val", time)
save_conf_mat(val_true, val_pred, model_name + "_val", time)
