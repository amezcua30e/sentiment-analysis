# sentiment-analysis
Multiclass sentiment classifier trained on 300K+ Amazon video game reviews using TF-IDF and Linear SVC. Includes custom preprocessing, model tuning, and benchmarking against logistic regression, CNN, and SMOTE. Achieved 66% validation accuracy despite class imbalance.


**Final Project**
The final project in this course is to predict the sentiment score based on Amazon reviews. To start, download the dataset from the following link. This dataset, released in 2018, is part of the larger Amazon review dataset. It includes various product reviews (ratings, text, helpfulness votes), metadata (such as descriptions, categories, price, brand, image features), and product links (e.g., “also viewed” and “also bought” graphs) organized by category. For this project, we'll focus on a subset specifically related to the Video Games category.

The dataset is formatted as one review per line in JSON, with fields including:

reviewerID - Reviewer's ID (e.g., A2SUAM1J3GNN3B)
asin - Product ID (e.g., 0000013714)
reviewerName - Reviewer's name
vote - Helpful votes received by the review
style - Dictionary of product metadata (e.g., Format is Hardcover)
reviewText - Text of the review
overall - Product rating
summary - Summary of the review
unixReviewTime - Review timestamp (Unix time)
reviewTime - Review timestamp (raw)
image - Images posted by users after receiving the product


**Project Goal**
The goal is to predict the overall rating using a test set provided as test.csv.gz. Your grade will be based on the performance of your classifier and the quality of your code.

**Tips and Guidelines**
Focus on Key Features: You are not required to use all information from the training data. Instead, focus primarily on the reviewText and overall fields. Examine reviewText in test.csv.gz and consider giving higher weights to similar observations in the training set.

**Model Selection:** Choose models strictly from those covered in class. You are not required to use deep learning models. A well-chosen text representation with linear models can often outperform poorly tuned deep learning models, as we've discussed in class.

**Submission Format: **Follow the submission example below to create submission.csv. Deviating from the specified format will result in a penalty.

**Notebook Requirements:** Ensure your notebook is error-free. If submitting via Colab, make sure to set the sharing permission to "anyone with the link."

**My work:**
Note: This whole project has been very difficult to work on due to the size of the data. Even sampling has been difficult. I will outline my whole process and what I tried below.


[ ]
from google.colab import drive
drive.mount('/content/drive')

Mounted at /content/drive

[ ]
import pandas as pd

# Load the JSON file
file_path = '/content/Video_Games_5.json'
df = pd.read_json(file_path, lines=True)

# Display the first few rows
df.head()



[ ]
train_data= df = pd.read_json(file_path, lines=True)[['reviewText', 'overall']]

*Overall, I ended up trying about 5-6 models. Most of them were not that successful in capturing the sentiments well. I got about 66% accuracy across all models. The computing times has been the biggest challenge here. I have been getting RAM crash after RAM crash because this system cannot handle it. Any small change I make to some code I have to wait about 15-20 additional minutes for all of the code to run again. Sampling has helped a little but not too much. I will explain each thing I tried, but in the end I ended up using Linear SVC as this was the most consistently accurate model. *


[ ]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Ensure the required NLTK corpora are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


test_data = pd.read_csv('/content/test.csv')
[nltk_data] Downloading package wordnet to /root/nltk_data...
[nltk_data] Downloading package omw-1.4 to /root/nltk_data...
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.

[ ]

test_data = test_data[['reviewText']]

Data Inspection

[ ]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display basic statistics for the 'overall' ratings
print("Training Data 'Overall' Distribution:")
print(train_data['overall'].value_counts())


# Visualize the distribution of scores in both datasets
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='overall', data=train_data, palette='viridis')
plt.title('Score Distribution in Training Data')
plt.xlabel('Score')
plt.ylabel('Frequency')


# Adding a column for text length to both datasets
train_data['text_length'] = train_data['reviewText'].apply(lambda x: len(str(x)))
test_data['text_length'] = test_data['reviewText'].apply(lambda x: len(str(x)))

# Plotting histograms of review text lengths
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_data['text_length'], bins=50, color='blue', kde=True)
plt.title('Review Text Length Distribution in Training Data')
plt.xlabel('Text Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(test_data['text_length'], bins=50, color='green', kde=True)
plt.title('Review Text Length Distribution in Test Data')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Boxplots of text length by score category
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='overall', y='text_length', data=train_data)
plt.title('Text Length by Score in Training Data')
plt.xlabel('Score')
plt.ylabel('Text Length')



Looking at these graphs tell us a couple of things: the data is imbalanced. Most of the scores are from 5s. About 3/5 scores are a 5. Imbalanced classes can lead to biased models that perform well on the majority class (score 5) but poorly on minority classes (scores 1, 2). The test and training data set look very similar with the text length distribution. Most of them are around the same length but they are some very big outliers.

Sampling Data
Because of extremely high computing times, the data will be sampled.


[ ]
from sklearn.model_selection import train_test_split

train_data, _ = train_test_split(train_data, test_size=0.9, stratify=train_data['overall'], random_state=42)


[ ]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display basic statistics for the 'overall' ratings
print("Training Data 'Overall' Distribution:")
print(train_data['overall'].value_counts())


# Visualize the distribution of scores in both datasets
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='overall', data=train_data, palette='viridis')
plt.title('Score Distribution in Training Data')
plt.xlabel('Score')
plt.ylabel('Frequency')


# Adding a column for text length to both datasets
train_data['text_length'] = train_data['reviewText'].apply(lambda x: len(str(x)))
test_data['text_length'] = test_data['reviewText'].apply(lambda x: len(str(x)))

# Plotting histograms of review text lengths
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_data['text_length'], bins=50, color='blue', kde=True)
plt.title('Review Text Length Distribution in Training Data')
plt.xlabel('Text Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(test_data['text_length'], bins=50, color='green', kde=True)
plt.title('Review Text Length Distribution in Test Data')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Boxplots of text length by score category
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='overall', y='text_length', data=train_data)
plt.title('Text Length by Score in Training Data')
plt.xlabel('Score')
plt.ylabel('Text Length')

Sampling looks consistent with the bigger data. Percentages of each number are the same.

Data Preprocessing
A quick inspection at the data shows us that there can be a lot of things for preprocessing that do make sense:

Lowercasing: Even though some words appear such as "BOOOM" or something like that, keeping the uppercase does not really make a difference, so it is better to lowercase it.

Tokenization: There is no reason unto why we would not tokenize in this context.

Removing excessive punctuation: Even though some sentences have something such as "it was great!!!" it still does not make a difference for our task.

Removing Stop Words: Words like "the," "is," "and" are very common but don't contribute much meaning in sentiment analysis. Some stopwords do so not everything will be removed.

Removing Excess Whitespace: Removing them ensures the data is formatted correctly and avoids errors during vectorization

Handling Contractions: Getting rid of contractions makes sentences clearer.

Getting rid of special characters: Special characters do not contribute to anything in this analysis.

8. Lemmatization was something I experimented with as I was not sure if it would be beneficial or not. In the end it was not. It took away 1% of the accuracy.


[ ]
import re  # For regular expressions
import nltk  # For downloading NLTK data
from nltk.corpus import stopwords  # For stopwords

# Ensure the required NLTK corpora are downloaded
nltk.download('stopwords')  # For stopwords

def preprocess_text(text, remove_stopwords=False):
    """
    Preprocess the text by lowercasing, handling negations, removing unwanted special characters,
    and optionally removing stopwords, without applying lemmatization.
    """
    if not isinstance(text, str):  # Handle non-string input gracefully
        return ""

    # Lowercase the text
    text = text.lower()

    # Handle negations by connecting them to the following word
    negation_pattern = re.compile(
        r'\b(not|no|never|nothing|nowhere|neither|nor|hardly|don\'t|can\'t|won\'t|isn\'t|aren\'t|didn\'t)\s([a-z]+)',
        re.IGNORECASE
    )
    text = negation_pattern.sub(lambda x: f"{x.group(1)}_{x.group(2)}", text)

    # Remove unwanted special characters while keeping !, ., ?, and ,
    text = re.sub(r"[@#$%^&*()_\[\]{};:<>|+=~`\"']", '', text)

    # Normalize multiple punctuation marks
    text = re.sub(r'([!.?,])\1+', r'\1', text)

    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords if required
    if remove_stopwords:
        custom_stopwords = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never', 'very', 'really', 'too'}
        text = " ".join(word for word in text.split() if word not in custom_stopwords)

    return text


# Apply preprocessing to dataframes
train_data['reviewText'] = train_data['reviewText'].fillna("").astype(str).apply(preprocess_text)
test_data['reviewText'] = test_data['reviewText'].fillna("").astype(str).apply(preprocess_text)

# Display a sample of the preprocessed data
print(train_data.head())
print(test_data.head())

[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
                                               reviewText  overall  \
246397  i am really liking this game. it has a cool st...        4   
229036  well, i recieved my game yesterday on the rele...        4   
153307                                          excellent        5   
424386                                               nice        4   
358437  wish they would do a little better in this ser...        3   

        text_length  
246397          473  
229036         2638  
153307            9  
424386            4  
358437          125  
                                          reviewText  text_length
0  two awesome games combined on 1 disc and a con...           98
1  great product to store a 3ds. i bought it for ...          123
2  it is exactly what was described would gladly ...           96
3  decent quality headphones for a decent price. ...          197
4       got for my son for christmas and he loves it           44
Models

[ ]
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data['reviewText'], train_data['overall'], test_size=0.2, random_state=42, stratify=train_data['overall'])



[ ]
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import RandomizedSearchCV





[ ]
from sklearn.model_selection import RandomizedSearchCV

# Set up the pipeline with TfidfVectorizer and LinearSVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', LinearSVC(dual=False, random_state=42))  # ensure dual is False as per your specification
])

# Define the parameter grid
param_distributions = {
    'tfidf__max_features': [10000, 20000, 30000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'svc__C': [0.1, 1, 10],  # including your specific value and other options to explore
    'svc__class_weight': ['balanced'],  # explicitly handle class imbalance
    'svc__max_iter': [2000],  # as specified
    'svc__loss': ['squared_hinge'],  # as specified
    'svc__penalty': ['l2']  # as specified
}

# Configure RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=3,  # Adjust based on how many iterations you wish to perform
    cv=2,  # Cross-validation folds
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy'
)

# Assume X_train, y_train have been defined and are ready for training
random_search.fit(X_train, y_train)

# Output the best parameters and the best score
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validated accuracy: ", random_search.best_score_)

# Evaluate the best model on validation data
best_model = random_search.best_estimator_
predictions = best_model.predict(X_val)
print(classification_report(y_val, predictions))
Fitting 2 folds for each of 3 candidates, totalling 6 fits
Best parameters found:  {'tfidf__ngram_range': (1, 2), 'tfidf__max_features': 10000, 'svc__penalty': 'l2', 'svc__max_iter': 2000, 'svc__loss': 'squared_hinge', 'svc__class_weight': 'balanced', 'svc__C': 0.1}
Best cross-validated accuracy:  0.6483859137679973
              precision    recall  f1-score   support

           1       0.48      0.66      0.56       618
           2       0.24      0.29      0.26       483
           3       0.34      0.36      0.35       983
           4       0.47      0.34      0.39      1873
           5       0.83      0.85      0.84      5995

    accuracy                           0.67      9952
   macro avg       0.47      0.50      0.48      9952
weighted avg       0.66      0.67      0.66      9952


[ ]
# Using the best estimator from RandomizedSearchCV
best_model = random_search.best_estimator_

# Predict the 'overall' scores for the test_df
test_data['overall'] = best_model.predict(test_data['reviewText'])

# Save the DataFrame to a CSV file
test_data.to_csv('/content/submission.csv', index=False)
What I Tried
Logistic Regression offered flexibility through regularization and class balancing, making it suitable for handling imbalanced datasets. I configured it with the liblinear solver for stability and increased the maximum iterations for better convergence. I also experimented with Multinomial Naive Bayes. Both are close or even the same to SVM but I could never get them higher than these scores, compared to SVM, which I once did.

[ ]
from sklearn.metrics import classification_report, accuracy_score
# Train Logistic Regression
logistic_classifier = LogisticRegression(
    C=1.0,                    # Regularization strength
    max_iter=2000,            # Increase max iterations
    solver='liblinear',       # Works well with smaller datasets
    class_weight='balanced',  # Handle class imbalance
    random_state=42           # Reproducibility
)
logistic_classifier.fit(X_train_tfidf, y_train)

# Evaluate Logistic Regression
y_val_pred_logistic = logistic_classifier.predict(X_val_tfidf)
print("Logistic Regression Classifier Performance:")
print(classification_report(y_val, y_val_pred_logistic))
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred_logistic):.2f}")

# Train Naive Bayes
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Evaluate Naive Bayes
y_val_pred_nb = naive_bayes_classifier.predict(X_val_tfidf)
print("Naive Bayes Classifier Performance:")
print(classification_report(y_val, y_val_pred_nb))
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred_nb):.2f}")
Logistic Regression Classifier Performance:
              precision    recall  f1-score   support

           1       0.54      0.64      0.59       631
           2       0.26      0.30      0.28       491
           3       0.34      0.35      0.35       983
           4       0.43      0.38      0.40      1890
           5       0.83      0.83      0.83      5957

    accuracy                           0.66      9952
   macro avg       0.48      0.50      0.49      9952
weighted avg       0.66      0.66      0.66      9952

Validation Accuracy: 0.66
Naive Bayes Classifier Performance:
              precision    recall  f1-score   support

           1       0.83      0.24      0.37       631
           2       0.00      0.00      0.00       491
           3       0.40      0.06      0.10       983
           4       0.36      0.28      0.32      1890
           5       0.68      0.94      0.79      5957

    accuracy                           0.63      9952
   macro avg       0.46      0.30      0.32      9952
weighted avg       0.57      0.63      0.57      9952

Validation Accuracy: 0.63
/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
To address class imbalance in my dataset, I tried to apply SMOTE (Synthetic Minority Oversampling Technique), which generates synthetic samples for underrepresented classes. Performance did not improve from this with any of the models.

[ ]
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
I tried using a Convolutional Neural Network (CNN). I tokenized the review text using Keras' Tokenizer, applied padding, and limited the vocabulary size to 10,000 words to manage computational costs. The model consisted of an Embedding Layer followed by a Conv1D layer for extracting key features from the text. A Global Max Pooling layer condensed the features, followed by Dense Layers for classification. I compiled the model using the Adam optimizer and trained it over 50 epochs. The accuracy did not go above 66%. I tried experimenting with the neurons and hidden layers but it did not improve.

[ ]
# Parameters
vocab_size = 10000  # Limit the vocabulary size
max_length = 100    # Max number of words per review
embedding_dim = 50  # Dimension of embedding space

# Tokenization and Padding
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)  # Use raw text data
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post', truncating='post')

[ ]


# Define the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Conv1D(64, 5, activation='relu', kernel_regularizer=l2(0.01)),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


[ ]
# Adjust labels
y_train_adjusted = y_train - 1
y_val_adjusted = y_val - 1

# Ensure labels are integers
y_train_adjusted = y_train_adjusted.astype(int)
y_val_adjusted = y_val_adjusted.astype(int)

# Verify unique values
print("Unique values in y_train_adjusted:", set(y_train_adjusted))
print("Unique values in y_val_adjusted:", set(y_val_adjusted))

history = model.fit(
    X_train_padded,
    y_train_adjusted,
    validation_data=(X_val_padded, y_val_adjusted),
    epochs=50,
    batch_size=32,
    )











[ ]
loss, accuracy = model.evaluate(X_val_padded, y_val_adjusted)
print(f"Validation Accuracy: {accuracy:.2f}")
3110/3110 ━━━━━━━━━━━━━━━━━━━━ 7s 2ms/step - accuracy: 0.6617 - loss: 0.9120
Validation Accuracy: 0.66
To enhance model performance, I tried to calculate cosine similarity between the TF-IDF representations of the training and test sets. By determining the maximum similarity for each training sample, I assigned sample weights based on these values. This approach emphasizes training samples most similar to the test data, helping the model focus on relevant patterns. In the end, there was little to no improvement, and this code most of the time killed my RAM depending on the size of the sample


[ ]
# Calculate cosine similarity between train and test sets for sample weighting

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarities = cosine_similarity(X_test_tfidf, X_train_tfidf)
sample_weights = cosine_similarities.max(axis=0).flatten()  # Use the maximum similarity as the weight for each sample
There was once or twice that I was able to get the SVM past ~66% but since I was trying to go higher, I forgot what I did to get there and I could not get there again. I am not sure why it has been very difficult to get past 66%, even with CNNs. I know the data is imbalanced but it did not do much when using SMOTE or other balancing methods. I tried asking ChatGPT for help with this but its solutions yielded results that were not satisfactory. I think using more advanced models might probably help with this. I think I did all of the correct preprocessing so I do not think it has to do with that.

