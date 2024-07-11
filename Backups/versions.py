import tensorflow as tf
# import torch
# import transformers
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
# import pandas as pd 
import keras
from keras.optimizers import Adam
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# print("we did it reddit")
# print(transformers.__version__)
# print(tf.__version__)
# print(keras.__version__)
