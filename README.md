# CODSOFT-TASK-03
# MOVIE GENRE CLASSIFICATION
# Importing Libraries
The necessary libraries for data manipulation, visualization, text processing, and machine learning are imported. This includes libraries like pandas, matplotlib, seaborn, nltk, and scikit-learn modules.

# Reading and Loading Data
Training Data: The training data is loaded from train_data.txt into a pandas DataFrame, with columns specified as 'Title', 'Genre', and 'Description'.
Test Data: The test data is loaded from test_data.txt into a pandas DataFrame, with columns specified as 'Id', 'Title', and 'Description'.
# Data Exploration
 Basic statistics and information about the training data are printed using describe() and info() methods to understand the dataset's structure and content.
 Data Visualization
Genre Distribution: Two plots are created to visualize the distribution of genres in the training data:
A count plot using seaborn's countplot() function.
A bar plot showing the counts of each genre using seaborn's barplot() function.
Text Length Distribution: A histogram is plotted to visualize the distribution of the lengths of the cleaned text in the training data.
# Downloading NLTK Data
Necessary NLTK datasets, such as 'stopwords' and 'punkt', are downloaded to support text preprocessing tasks.
# Text Preprocessing
Stopwords and Stemming: NLTK's LancasterStemmer and stop words for English are initialized.
Text Cleaning Function: A function clean_text() is defined to preprocess text by:
Converting to lowercase.
Removing Twitter handles, URLs, non-alphabetic characters, punctuation, and stopwords.
Tokenizing and stemming words.
Applying Text Cleaning: The clean_text() function is applied to the 'Description' column of both the training and test datasets to create a new column 'Text_cleaning'.
# Feature Extraction
TF-IDF Vectorization: A TfidfVectorizer is initialized to transform the cleaned text data into numerical features:
The training data is fit and transformed to create X_train.
The test data is transformed using the same vectorizer to create X_test.
# Data Splitting
The dataset is split into training and validation sets using train_test_split() with 80% of the data for training and 20% for validation.
# Model Training
Multinomial Naive Bayes Classifier: A MultinomialNB classifier is initialized and trained using the training data (X_train and y_train).
# Model Evaluation
Validation Predictions: The trained model makes predictions on the validation set.
Performance Metrics:
Accuracy Score: The accuracy of the model on the validation set is calculated and printed.
Classification Report: A detailed classification report is generated, showing precision, recall, F1-score, and support for each genre.
# Test Data Prediction
Test Data Predictions: The trained model is used to predict genres for the test data (X_test).
Saving Predictions: The test data DataFrame is updated with the predicted genres and saved to a CSV file named 'predicted_genres.csv'.
# Displaying Results
The test data DataFrame with the predicted genres is printed to display the results.
