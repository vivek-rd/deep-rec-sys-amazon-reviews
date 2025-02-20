# %%
import pandas as pd
import numpy as np
import gzip
import os
import re
import nltk
import tensorflow as tf, keras
import matplotlib.pyplot as plt
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %%
nltk.download('stopwords')
nltk.download('wordnet')

# %%
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Appliances",
    split="full",
    # streaming=True,  # Enables streaming mode (no large file download)
    trust_remote_code=True
)

# %%
# Select only essential columns
essential_columns = ["user_id", "asin", "text", "rating", "timestamp"]
filtered_dataset = dataset.remove_columns([col for col in dataset.column_names if col not in essential_columns])

# %%
# Collect a sample of 100k 
sample_data = []
for i, sample in enumerate(filtered_dataset):
    if i >= 100000:
        break
    sample_data.append(sample)

# Convert to Pandas DataFrame
df = pd.DataFrame(sample_data)

# %%
df.to_csv("appliances_reviews.csv", header=True, index=False, encoding='utf-8')

# %%
# Data Preprocessing

# According to DeepCoNN, user reviews and item reviews are expected to be inputs for 2 different NNs

user_reviews = df.groupby("user_id")["text"].apply(lambda x: " ".join(x)).reset_index()
user_reviews.rename(columns={"text": "user_review"}, inplace=True)

item_reviews = df.groupby("asin")["text"].apply(lambda x: " ".join(x)).reset_index()
item_reviews.rename(columns={"text": "item_review"}, inplace=True)

df = df.merge(user_reviews, on="user_id", how="left")
df = df.merge(item_reviews, on="asin", how="left")

# %%
df.head()

# %%
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if isinstance(text, str):  # Check if text is valid
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^a-z0-9]", " ", text)  # Remove special characters
        text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatization
        return text
    return ""


df["user_review"] = df["user_review"].apply(clean_text)
df["item_review"] = df["item_review"].apply(clean_text)

# Save cleaned dataset
df.to_csv("amazon_appliances_cleaned.csv", index=False)

# %%
df.head()

# %%
# Convert user id and item id to numerical indices as NNs don't support alphanumeric ASINs

user2idx = {user: idx for idx, user in enumerate(df["user_id"].unique())}
item2idx = {item: idx for idx, item in enumerate(df["asin"].unique())}

df["user_id"] = df["user_id"].map(user2idx)
df["asin"] = df["asin"].map(item2idx)

df.to_csv("amazon_appliances_final.csv", index=False)

# %%
# Set parameters
MAX_VOCAB_SIZE = 50000  # Maximum number of words in the vocabulary
MAX_SEQUENCE_LENGTH = 300  # Maximum number of words per review

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df["user_review"].tolist() + df["item_review"].tolist())

# Convert text to sequences
df["user_review_seq"] = tokenizer.texts_to_sequences(df["user_review"])
df["item_review_seq"] = tokenizer.texts_to_sequences(df["item_review"])

# Pad sequences
df["user_review_seq"] = list(pad_sequences(df["user_review_seq"], maxlen=MAX_SEQUENCE_LENGTH, padding="post"))
df["item_review_seq"] = list(pad_sequences(df["item_review_seq"], maxlen=MAX_SEQUENCE_LENGTH, padding="post"))

# Save tokenized dataset
df.to_csv("amazon_appliances_tokenized.csv", index=False)

# %%
# Model Parameters
EMBEDDING_DIM = 300  # Word embedding dimension
FILTERS = 100  # Number of CNN filters
KERNEL_SIZE = 3  # Convolution window size
DENSE_UNITS = 128  # Fully connected layer units

def cnn_block(input_layer):
    x = Embedding(input_dim = MAX_VOCAB_SIZE, output_dim = EMBEDDING_DIM, input_length = MAX_SEQUENCE_LENGTH)(input_layer)
    x = Conv1D(filters=FILTERS, kernel_size = KERNEL_SIZE, activation="relu")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    return x

user_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), name="user_input")
user_output = cnn_block(user_input)

item_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), name="item_input")
item_output = cnn_block(item_input)

merged = Concatenate()([user_output, item_output])
merged = Dense(DENSE_UNITS, activation="relu")(merged)
rating_prediction = Dense(1, activation="linear")(merged)

deepconn_model = Model(inputs=[user_input, item_input], outputs=rating_prediction)

deepconn_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

deepconn_model.summary()

# %%
# Train the model

# df["user_review_seq"] = df["user_review_seq"].apply(lambda x: np.array(eval(x)))
# df["item_review_seq"] = df["item_review_seq"].apply(lambda x: np.array(eval(x)))

X_user = np.array(df["user_review_seq"].tolist()) 
X_item = np.array(df["item_review_seq"].tolist()) 
y = np.array(df["rating"])  

# %%
# Split the Data

X_user_train, X_user_test, X_item_train, X_item_test, y_train, y_test = train_test_split(
    X_user, X_item, y, test_size=0.2, random_state=42
)

# %%
history = deepconn_model.fit(
    [X_user_train, X_item_train], y_train,
    validation_data = ([X_user_test, X_item_test], y_test),
    epochs=5,
    batch_size=128,
    verbose=1
)

deepconn_model.save("deepconn_amazon_appliances.h5")

# %%
def round_to_nearest_half(pred):
    return round(pred * 2)/2

# %%
# Model Evaluation

y_pred = deepconn_model.predict([X_user_test, X_item_test])
y_pred = np.array(list(map(lambda x: round_to_nearest_half(x), y_pred[:, 0])))

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


# %%
# Plot the Loss Curve

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training Progress (DeepCoNN)")
plt.legend()
plt.show()


