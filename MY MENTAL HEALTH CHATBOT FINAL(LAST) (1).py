#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


def evaluate_model_performance(model, X_test, y_test):

    y_pred = model.predict(X_test)
    
 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1


# In[3]:


# Load dataset from JSON file
with open('intents.json') as file:
    intents_data = json.load(file)

# Preprocess data
patterns = []
tags = []
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Create DataFrame
df = pd.DataFrame({'patterns': patterns, 'tags': tags})

# Label encode target variable
label_encoder = LabelEncoder()
df['encoded_tags'] = label_encoder.fit_transform(df['tags'])

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['patterns'])
sequences = tokenizer.texts_to_sequences(df['patterns'])
max_sequence_length = max([len(sequence) for sequence in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')


# In[4]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['encoded_tags'], test_size=0.2, random_state=42)


# In[5]:


# Train SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)


# In[6]:


# Train KNN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)


# In[7]:


vocab_size = len(tokenizer.word_index) + 1
cnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(len(label_encoder.classes_), activation='softmax')
])


cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


# In[8]:


rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length),
    LSTM(128),
    Dense(len(label_encoder.classes_), activation='softmax')
])


rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


# In[9]:


svm_accuracy = svm_model.score(X_test, y_test)


knn_accuracy = knn_model.score(X_test, y_test)


cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)


rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)


# In[10]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predictions for SVM model
svm_predictions = svm_model.predict(X_test)

svm_model = SVC(class_weight='balanced')
svm_model.fit(X_train, y_train)

# Compute evaluation metrics for SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Predictions for KNN model
knn_predictions = knn_model.predict(X_test)
# Compute evaluation metrics for KNN
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

# Print evaluation metrics
print("SVM Model Metrics:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)

print("\nKNN Model Metrics:")
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)



# In[11]:


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Assuming X_test is already preprocessed and properly shaped for CNN prediction
# Get predicted probabilities from CNN model
cnn_pred_prob = cnn_model.predict(X_test)

# Get predicted class labels for CNN
cnn_pred = np.argmax(cnn_pred_prob, axis=1)

# Compute precision, recall, F1-score, and accuracy for CNN model
cnn_precision = precision_score(y_test, cnn_pred, average='weighted')
cnn_recall = recall_score(y_test, cnn_pred, average='weighted')
cnn_f1 = f1_score(y_test, cnn_pred, average='weighted')
cnn_accuracy = accuracy_score(y_test, cnn_pred)

print("\nCNN Model Metrics:")
print("Accuracy:", cnn_accuracy)
print("Precision:", cnn_precision)
print("Recall:", cnn_recall)
print("F1 Score:", cnn_f1)


# In[12]:


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Assuming X_test is already preprocessed and properly shaped for RNN prediction
# Get predicted probabilities from RNN model
rnn_pred_prob = rnn_model.predict(X_test)

# Get predicted class labels for RNN
rnn_pred = np.argmax(rnn_pred_prob, axis=1)

# Compute precision, recall, F1-score, and accuracy for RNN model
rnn_precision = precision_score(y_test, rnn_pred, average='weighted')
rnn_recall = recall_score(y_test, rnn_pred, average='weighted')
rnn_f1 = f1_score(y_test, rnn_pred, average='weighted')
rnn_accuracy = accuracy_score(y_test, rnn_pred)

print("\nRNN Model Metrics:")
print("Accuracy:", rnn_accuracy)
print("Precision:", rnn_precision)
print("Recall:", rnn_recall)
print("F1 Score:", rnn_f1)


# In[13]:


import pandas as pd
import plotly.graph_objects as go

# DataFrame to store the performance metrics
metrics_data = {
    "Model": ["SVM", "CNN", "RNN","KNN"],
    "Accuracy": [svm_accuracy, cnn_accuracy, rnn_accuracy,knn_accuracy],
    "Precision": [svm_precision, cnn_precision, rnn_precision,knn_precision],
    "Recall": [svm_recall, cnn_recall, rnn_recall,knn_recall],
    "F1 Score": [svm_f1, cnn_f1, rnn_f1,knn_f1]
}
metrics_df = pd.DataFrame(metrics_data)


fig = go.Figure(data=[go.Table(
    header=dict(values=list(metrics_df.columns),
                fill_color='aqua',
                align='left'),
    cells=dict(values=[metrics_df.Model, metrics_df.Accuracy, metrics_df.Precision, metrics_df.Recall, metrics_df["F1 Score"]],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(title="Model Performance Comparison",
                  title_font_size=20,
                  title_x=0.5)

fig.show()


# In[14]:


import plotly.graph_objects as go

models = ['SVM', 'CNN', 'RNN', 'KNN']

# Model performance metrics
accuracy = [svm_accuracy, cnn_accuracy, rnn_accuracy, knn_accuracy]
precision = [svm_precision, cnn_precision, rnn_precision, knn_precision]
recall = [svm_recall, cnn_recall, rnn_recall, knn_recall]
f1_score = [svm_f1, cnn_f1, rnn_f1, knn_f1]

#subplots
fig = go.Figure(data=[
    go.Bar(name='Accuracy', x=models, y=accuracy),
    go.Bar(name='Precision', x=models, y=precision),
    go.Bar(name='Recall', x=models, y=recall),
    go.Bar(name='F1 Score', x=models, y=f1_score)
])


fig.update_layout(barmode='group', title='Model Performance Metrics', xaxis_title='Model', yaxis_title='Score')

fig.show()


# In[15]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Concatenate all patterns into a single string
all_patterns = ' '.join(df['patterns'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_patterns)

wordcloud_array = wordcloud.to_array()

fig = go.Figure()

fig.add_trace(go.Image(z=wordcloud_array))

fig.update_layout(title='Word Cloud')

fig.show()


# In[16]:


import plotly.express as px

word_lengths = df['patterns'].apply(lambda x: len(x.split()))

fig = px.line(x=range(len(word_lengths)), y=word_lengths, title='Line Plot of Word Lengths')
fig.update_layout(xaxis_title='Index', yaxis_title='Word Length')
fig.show()


# In[17]:


pattern_lengths = df['patterns'].apply(lambda x: len(x))

fig = px.histogram(pattern_lengths, x='patterns', nbins=20, title='Distribution of Pattern Lengths')
fig.update_traces(marker=dict(color='pink', line=dict(color='black', width=1)))
fig.show()


# In[18]:


# Create scatter plot
fig = px.scatter(x=range(len(word_lengths)), y=word_lengths, title='Scatter Plot of Word Lengths')
fig.update_layout(xaxis_title='Index', yaxis_title='Word Length')
fig.show()


# In[19]:


# Create box plot
fig = px.box(y=pattern_lengths, title='Box Plot of Pattern Lengths')
fig.show()


# In[20]:


import plotly.express as px

# Compute pattern lengths and add as a new column to the DataFrame
df['pattern_length'] = df['patterns'].apply(lambda x: len(x.split()))

# Create a box plot
fig = px.box(df, x='tags', y='pattern_length', title='Distribution of Pattern Lengths Across Intents')

# Update layout
fig.update_layout(xaxis_title='Intent', yaxis_title='Pattern Length')

# Show the plot
fig.show()


# In[21]:


from PIL import Image, ImageDraw

def create_gradient_image(width, height, color1, color2):
    gradient = Image.new('RGB', (width, height), color1)
    draw = ImageDraw.Draw(gradient)
    for i in range(height):
        r = color1[0] + (color2[0] - color1[0]) * i / height
        g = color1[1] + (color2[1] - color1[1]) * i / height
        b = color1[2] + (color2[2] - color1[2]) * i / height
        draw.line((0, i, width, i), fill=(int(r), int(g), int(b)))
    return gradient

gradient_image = create_gradient_image(400, 500, (255, 215, 0), (255, 192, 203))
gradient_image.save("gradient_background.png")


# In[ ]:


import tkinter as tk
from tkinter import scrolledtext, END
import random
import json

class Chatbot:
    def __init__(self, dataset_path):
        self.dataset = self.load_dataset(dataset_path)
        self.colors = ["#FFB6C1", "#FFFACD", "#E6E6FA", "#FFDAB9", "#98FB98", "#87CEEB"]
        self.current_color_index = 0

    def load_dataset(self, dataset_path):
        with open(dataset_path, 'r') as file:
            data = json.load(file)
        return data['intents']

    def get_response(self, user_input):
        matched_responses = []
        for intent in self.dataset:
            for pattern in intent['patterns']:
                if user_input.lower() in pattern.lower():
                    matched_responses.extend(intent['responses'])
        if matched_responses:
            return random.choice(matched_responses)
        else:
            return "oh okay!!I am here for you my friend!!"

    def start_chat(self):
       
        self.root = tk.Tk()
        self.root.title("Chatbot")
        self.root.geometry("400x500")
        self.update_background_color()  

        
        self.chat_history = scrolledtext.ScrolledText(self.root, width=40, height=20, wrap=tk.WORD, bg="#FFFFFF", 
                                                      fg="#000000", font=("Comic Sans MS", 12))
        self.chat_history.grid(row=0, column=0, padx=10, pady=10, columnspan=2, sticky="nsew")

        
        self.input_box = tk.Entry(self.root, width=30, bg="#FFFFFF", fg="#000000", font=("Comic Sans MS", 12))
        self.input_box.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

       
        self.send_button = tk.Button(self.root, text="Send", bg="#FF6347", fg="#FFFFFF", font=("Comic Sans MS", 12),
                                     command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

       
        self.display_message("Mannmitra", "Hello my friend. how are you? ^-^ ")

        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Run GUI
        self.root.mainloop()

    def update_background_color(self):
        self.root.configure(bg=self.colors[self.current_color_index])
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.root.after(2000, self.update_background_color)

    def send_message(self):
        user_input = self.input_box.get()
        self.display_message("You", user_input)
        response = self.get_response(user_input)
        self.display_message("Bot", response)
        self.input_box.delete(0, END)

    def display_message(self, sender, message):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(END, f"{sender}: {message}\n")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.see(END)

dataset_path = 'intents.json'  
chatbot = Chatbot(dataset_path)
chatbot.start_chat()


# In[ ]:


import openai
print("OpenAI package is installed and ready to use.")


# In[ ]:




