import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Sample dataset (simulate spam detection)
data = {
    'text': [
        "Congratulations! You've won a $1,000 Walmart gift card. Click to claim now!",
        "Hey, are we still on for lunch today?",
        "Lowest price on meds, guaranteed! Visit our online pharmacy now!",
        "Meeting rescheduled to 2PM. Let me know if that works.",
        "Win big prizes, just send us your bank details.",
        "Sure, I'll review the document and get back to you."
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Step 2: Preprocess text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered)

df['cleaned_text'] = df['text'].apply(preprocess)

# Step 3: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train a classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
