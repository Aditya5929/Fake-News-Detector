import pandas as pd
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Verify if the dataset files exist in the current directory
if not os.path.exists('True.csv') or not os.path.exists('Fake.csv'):
    print("❌ Error: 'True.csv' or 'Fake.csv' not found in folder! Please check your dataset path.")
    exit()

print(" Loading Dataset Files...")
df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

# 2. Assign ground truth labels (Real = 0, Fake = 1)
df_true['label'] = 0
df_fake['label'] = 1

# 3. Combine both datasets into a single dataframe
print(" Merging Real and Fake datasets...")
df = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)

# Retain only the required columns for training
df = df[['text', 'label']]

# Shuffle the combined dataset to ensure uniform distribution during training
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#  Text Preprocessing Pipeline (Cleaning text data for better accuracy)
def clean_text(text):
    if isinstance(text, str):
        text = text.lower() # Convert text to lowercase
        text = re.sub(r'\[.*?\]', '', text) # Remove text inside square brackets
        text = re.sub(r'\W', ' ', text) # Remove special characters and punctuation
        text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs/Hyperlinks
        text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
        text = re.sub(r'\n', ' ', text) # Remove newline characters
        text = re.sub(r'\w*\d\w*', '', text) # Remove words that contain numbers
        return text
    return ""

print("🧹 Preprocessing text dataset rows (Please wait)...")
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

# 4. Perform Train-Test Split (80% for Training, 20% for Testing/Evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(" Transforming Text Data using TF-IDF Vectorizer...")
# max_features limits vocabulary size to optimize performance and memory footprint
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) 
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(" Training Logistic Regression Classifier...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

#  Evaluation Phase
predictions = model.predict(X_test_vectorized)
print("\n=====  MODEL PERFORMANCE EVALUATION REPORT =====")
print(f"Overall Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Classification Metrics:")
print(classification_report(y_test, predictions)) # Displays Precision, Recall, and F1-score
print("==================================================")

# 5. Serialize and save the trained model weights and vectorizer configuration
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("\n Success! Model weights and vectorizer configuration saved successfully!")