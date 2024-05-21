#!/usr/bin/env python
# coding: utf-8

# In[4]:


import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Uncomment these lines if you haven't downloaded the nltk data before
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load sample data
data = pd.read_csv("C:/Users/user/OneDrive/Desktop/DATASETS/Grammar Correction.csv")  # Replace with actual dataset path

# Print column names to verify
print("Column names in the dataset:", data.columns)

# Check if the necessary columns are present
if 'Ungrammatical Statement' not in data.columns or 'Error Type' not in data.columns:
    raise KeyError("The dataset must contain 'Ungrammatical_Statement' and 'Error_Type' columns")

# Preprocessing function
def preprocess_text(Ungrammatical_Statement):
    # Lowercase conversion
    # Ungrammatical_Statement = Ungrammatical_Statement.lower()
    # Remove special characters
    Ungrammatical_Statement = re.sub(r'\W', ' ', Ungrammatical_Statement)
    # Tokenization
    words = word_tokenize(Ungrammatical_Statement)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply preprocessing
data['cleaned_text'] = data['Ungrammatical Statement'].apply(preprocess_text)

# Label encoding (assuming binary classification for errors)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Error Type'])

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_text'])

# Apply K-means clustering
num_clusters = 10  # Set the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Add cluster labels as features
X = np.hstack((tfidf_matrix.toarray(), data['cluster'].values.reshape(-1, 1)))
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", random_search.best_params_)

# Get the best model
best_rf_model = random_search.best_estimator_

# Evaluate the best model on the test data
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Model evaluation:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")



# In[8]:


# Load the pre-trained T5 model and tokenizer fine-tuned for grammar correction
model_name = 'vennify/t5-base-grammar-correction'  # Known model for grammar correction
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to correct grammatical errors using the T5 model
def correct_grammar(text, model, tokenizer):
    input_text = "gec: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Function to detect errors and correct the sentence
def detect_and_correct_errors(text, rf_model, vectorizer, kmeans, t5_model, t5_tokenizer, label_encoder):
    # Preprocess the text for error detection
    cleaned_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    text_cluster = kmeans.predict(text_tfidf)[0]
    text_features = np.hstack((text_tfidf.toarray(), np.array([[text_cluster]])))
    # Detect the error type
    predicted_label = rf_model.predict(text_features)
    # Check if the sentence has no errors based on some threshold
    if predicted_label == 0:  # Assuming 0 is the label for "no error"
        return "No error in the sentence", text
    error_type = label_encoder.inverse_transform(predicted_label)[0]
    # Correct the sentence using the T5 model
    corrected_sentence = correct_grammar(text, t5_model, t5_tokenizer)
    return error_type, corrected_sentence

# Take user input for testing the model
while True:
    user_input = input("Enter a sentence for error detection and correction (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    detected_error_type, corrected_sentence = detect_and_correct_errors(user_input, best_rf_model, tfidf_vectorizer, kmeans, t5_model, t5_tokenizer, label_encoder)
    print(f"Detected Error Type: {detected_error_type}")
    print(f"Corrected Sentence: {corrected_sentence}")


# In[ ]:




