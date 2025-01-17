import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Preprocess text
def preprocess(text):
    """Cleans text by converting to lowercase and removing special characters."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text

# Create a dataset with additional neutral examples
data = pd.DataFrame({
    "review": [
        "I love this product, it works great!",
        "This is the best experience I have ever had.",
        "Amazing quality and very fast shipping.",
        "I had an amazing experience, very happy.",
        "Really good product.",
        "I hate this product. Terrible experience.",
        "Disappointing product, very bad.",
        "Not worth the price, bad performance.",
        "The product arrived in a box with standard delivery.",
        "I used the product as expected and it works fine.",
        "The weather was okay during my trip.",
        "I went to the store to buy groceries today.",
        "The meeting went fine with no complaints from anyone."
    ],
    "sentiment": [
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative",
        "Neutral", "Neutral", "Neutral", "Neutral", "Neutral"
    ]
})

# Preprocess the reviews
data['review_cleaned'] = data['review'].apply(preprocess)

# Convert sentiment to numerical labels for classification
label_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
data["sentiment_label"] = data["sentiment"].map(label_mapping)

# Split data into train and test sets
X = data['review_cleaned']
y = data['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a machine learning model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict sentiment on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Add predictions to the original data
data['predicted_sentiment'] = model.predict(vectorizer.transform(data['review_cleaned']))

# Map the numerical predictions back to sentiment labels
data['predicted_sentiment'] = data['predicted_sentiment'].map({1: "Positive", 0: "Neutral", -1: "Negative"})

# Display the results
print("\nPredicted Sentiment:\n", data[['review', 'sentiment', 'predicted_sentiment']])

# Plotting sentiment distribution
sns.countplot(x="predicted_sentiment", data=data, hue="predicted_sentiment", palette="viridis", legend=False)
plt.title("Predicted Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
