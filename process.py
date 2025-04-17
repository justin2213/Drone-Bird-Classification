
import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16, VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess

def load_images(dir, limit=100, image_size=(224, 224)):
    images = []
    labels = []
    count = 0
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit // 2]:
            if count >= limit:
                break
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)
                images.append(image)
                labels.append(root.split("/")[-1])
                count += 1
        if count >= limit:
            break
    unique = list(set(labels))
    print("Unique labels found:", unique)
    return np.array(images), labels

def train_test_val(dir, limit=100, image_size=(224, 224)):
    test_limit = int(limit * 0.15)
    val_limit = int(limit * 0.15)
    print("Loading training data")
    train = load_images(dir=os.path.join(dir, "train"), limit=limit, image_size=image_size)
    print("Loading testing data")
    test = load_images(dir=os.path.join(dir, "test"), limit=test_limit, image_size=image_size)
    print("Loading validation data")
    val = load_images(dir=os.path.join(dir, "valid"), limit=val_limit, image_size=image_size)
    return train, test, val

def normalize_and_encode_train(data):
    X, y = data
    X = X / 255.0
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    y_enc = y_enc.reshape(-1, 1)
    return X, y_enc, label_encoder

def normalize_and_encode_transform(data, label_encoder):
    X, y = data
    X = X / 255.0
    y_enc = label_encoder.transform(y)
    y_enc = y_enc.reshape(-1, 1)
    return X, y_enc

def prepare_datasets(data_path, limit=100, image_size=(224, 224)):
    train, test, val = train_test_val(data_path, limit, image_size)
    X_train, y_train, label_encoder = normalize_and_encode_train(train)
    X_test, y_test = normalize_and_encode_transform(test, label_encoder)
    X_val, y_val = normalize_and_encode_transform(val, label_encoder)
    return X_train, y_train, X_test, y_test, X_val, y_val, label_encoder

def undo_encoding(y_encoded, label_encoder):
    return label_encoder.inverse_transform(y_encoded.flatten())

def get_predictions(y_test_pred, y_test, y_val_pred, y_val, label_encoder):
    y_test_pred_int = (y_test_pred > 0.5).astype(int)
    y_val_pred_int = (y_val_pred > 0.5).astype(int)
    
    y_test_true_int = y_test.flatten()
    y_val_true_int = y_val.flatten()
    
    test_pred = label_encoder.inverse_transform(y_test_pred_int.flatten())
    test_true = label_encoder.inverse_transform(y_test_true_int)
    val_pred = label_encoder.inverse_transform(y_val_pred_int.flatten())
    val_true = label_encoder.inverse_transform(y_val_true_int)
    
    return test_pred, test_true, val_pred, val_true

def encode_labels_train(y):
    label_encoder = LabelEncoder()
    y_num = label_encoder.fit_transform(y)
    return y_num, label_encoder

def encode_labels_transform(y, label_encoder):
    return label_encoder.transform(y)

def undo_label_encoding(y_num, label_encoder):
    return label_encoder.inverse_transform(y_num)

def load_images_and_extract_features(dir, limit=100):
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    vgg16_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    vgg19_model = VGG19(weights='imagenet', include_top=False, pooling='avg')

    images = []
    labels = []
    count = 0

    for root, dirs, files in os.walk(dir):
        for file in files[0:limit // 2]:
            if count >= limit:
                break
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                original_img = cv2.imread(image_path)
                rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                # ResNet (224x224)
                resnet_img = cv2.resize(rgb_img, (224, 224))
                resnet_img = resnet_preprocess(resnet_img)
                resnet_img = np.expand_dims(resnet_img, axis=0)
                resnet_features = resnet_model.predict(resnet_img).flatten()

                # Inception (299x299)
                inception_img = cv2.resize(rgb_img, (299, 299))
                inception_img = inception_preprocess(inception_img)
                inception_img = np.expand_dims(inception_img, axis=0)
                inception_features = inception_model.predict(inception_img).flatten()

                # VGG16 (224x224)
                vgg16_img = cv2.resize(rgb_img, (224, 224))
                vgg16_img = vgg16_preprocess(vgg16_img)
                vgg16_img = np.expand_dims(vgg16_img, axis=0)
                vgg16_features = vgg16_model.predict(vgg16_img).flatten()

                # VGG19 (224x224)
                vgg19_img = cv2.resize(rgb_img, (224, 224))
                vgg19_img = vgg19_preprocess(vgg19_img)
                vgg19_img = np.expand_dims(vgg19_img, axis=0)
                vgg19_features = vgg19_model.predict(vgg19_img).flatten()

                # Determine max length for zero-padding
                max_length = max(len(resnet_features), len(inception_features),
                                 len(vgg16_features), len(vgg19_features))

                resnet_features    = np.pad(resnet_features,    (0, max_length - len(resnet_features)),    'constant')
                inception_features = np.pad(inception_features, (0, max_length - len(inception_features)), 'constant')
                vgg16_features     = np.pad(vgg16_features,     (0, max_length - len(vgg16_features)),     'constant')
                vgg19_features     = np.pad(vgg19_features,     (0, max_length - len(vgg19_features)),     'constant')

                # Fuse the features
                fused_features = np.concatenate(
                    [resnet_features, inception_features, vgg16_features, vgg19_features],
                    axis=0
                )

                images.append(fused_features)
                labels.append(root.split("/")[-1])
                count += 1

        if count >= limit:
            break

    unique = list(set(labels))
    print("Labels found in {}: {}".format(dir, unique))
    return np.array(images), labels
