# Text-Base
This is my first repository
# 📧 Spam Message Detection Using Machine Learning

This project focuses on building a **text classification system** to detect spam messages using Natural Language Processing (NLP) and Machine Learning. It leverages a dataset of SMS messages labeled as **"ham" (not spam)** or **"spam"**, and includes preprocessing, feature extraction, model training, evaluation, and deployment.


## 🗂️ Dataset Overview

* **Source:** 'spam.csv'
* 
* **Preprocessing:**

  * Removed unnecessary columns ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4')
  * Renamed 'v1' → target, v2 → 'text'
  * Encoded labels: "ham" → 0, "spam" → 1
  * Removed duplicate messages


## 🔎 Exploratory Data Analysis (EDA)

* **Feature Engineering:**

  * Number of characters
  * Number of words
  * Number of sentences

* **Visualizations:**

  * Pie chart showing class distribution
  * Histograms comparing message lengths in spam vs ham


## 🧼 Text Preprocessing (NLP)

Custom function: transform_text()

Steps included:

1. Lowercasing
2. Tokenization (`nltk.word_tokenize`)
3. Removing special characters
4. Removing stopwords and punctuation
5. Stemming using 'PorterStemmer'

> Example:

python
transform_text('Hi there! How are you today?') → "hi there today"

##  WordCloud Visualization

* Created WordClouds for both spam and ham messages
* Helps identify frequently used keywords in each category



## Feature Extraction

Used **TF-IDF Vectorization**:

  python
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
```

* Extracts up to 3000 most relevant terms
* Converts text into numerical feature vectors


## Model Training

Split data:

* X_train, X_test, y_train, y_test (80/20 split)

Trained and evaluated:

* GaussianNB
* MultinomialNB ✅ (best results)
* BernoulliNB

Metrics used:

* **Accuracy**
* **Confusion Matrix**
* **Precision Score**

> Example Result:

  python
MultinomialNB
Accuracy: ~0.97
Precision: ~0.95


## 📂 Model Export

Serialized and saved:

* Trained model → model.pkl
* Vectorizer → vectorizer.pkl

```python
import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))


These can be used later for inference in a web app (e.g., Flask or Streamlit).


## ✅ Dependencies

Make sure to install:

pip install numpy pandas matplotlib seaborn nltk wordcloud scikit-learn


## 📂 Folder Structure

```
spam-detection/
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── spam_detection.ipynb
└── README.md




---

## 👨‍💻 Author

**\[Your Name]** – 
GitHub: [Sunny Wazeer](https://github.com/Sunny Wazeer)
