#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import torch
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import pandas as pd 
import keras
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


# get_ipython().system('CUDA_VISIBLE_DEVICES=5')


# In[3]:


from transformers import CLIPTokenizer,AutoTokenizer
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')


# In[4]:


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)


# In[5]:


label_encoder = LabelEncoder()


# In[6]:


df = pd.read_csv(f'/home/rpierson/Topic_Files/topic_0.csv')
df['Priority'] = label_encoder.fit_transform(df['Priority'])

for x in range(len(df)):
    if pd.isna(df.iloc[x]["Combined_Text"]):
        df.at[x, "Combined_Text"] = " "
        
count = 0
for x in range(len(df)):
    if pd.isna(df.iloc[x]["Combined_Text"]):
        count += count
        
count


# In[7]:


df.tail()


# In[8]:


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


# In[9]:


def tokenize_data(texts, labels, tokenizer, max_len):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_len)
    return tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels
    ))


# In[10]:


train_dataset = tokenize_data(train_df.Combined_Text, train_df.Priority, tokenizer, max_len=128)
val_dataset = tokenize_data(val_df.Combined_Text, val_df.Priority, tokenizer, max_len=128)


# In[11]:


train_dataset = train_dataset.shuffle(len(train_df)).batch(16)
val_dataset = val_dataset.batch(16)


# In[12]:


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]


# In[13]:


if torch.cuda.is_available():
    print("CUDA is available. Number of GPUs:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(5))
else:
    print("CUDA is not available.")
torch.cuda.set_device(torch.device("cuda:5"))


# In[14]:


# TensorFlow GPU configuration
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if physical_devices:
#    try:
#        for device in physical_devices:
#            tf.config.experimental.set_memory_growth(device, True)
#        print(f"LOG: TensorFlow GPU devices: {physical_devices}.", flush=True)
#    except RuntimeError as exception:
#        print(f"LOG: TensorFlow GPU devices: {physical_devices}.", flush=True)
#        print(f"LOG: TensorFlow GPU configuration error: {exception}", flush=True)
#else:
#    print("ERROR: No TensorFlow GPU devices found.", flush=True)


# In[15]:


strategy = tf.distribute.MirroredStrategy()


# In[16]:


model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[ ]:


model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15
)


# In[ ]:





# In[ ]:




