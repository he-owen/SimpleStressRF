# Standard includes
import pickle
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_recall_fscore_support, classification_report


# Setup pre-processing definitions
from sklearn.model_selection import train_test_split

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# Helper Functions
def clean(doc):
    stop_free = " ".join([i for i in doc.split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
def add_one_to_data(attribute):
    # Categories in data set start at -1, ex. Data Values: -1 0 1 2 3
    # This adds 1 to all data values in the specified attribute column, converts to numeric labels as well
    df[attribute] = pd.to_numeric(df[attribute], errors='coerce').add(1).astype(int)

def count_num_of_data_per_category(attribute):
    CategoryLabels = list(df[attribute])
    alist = []
    for i in range(male_num_of_categories[attribute]):
        alist.append(CategoryLabels.count(i))
    print("Training Attribute: " + attribute)
    return alist

def print_data_distribution(category_counts):
    total_categories = len(category_counts)
    total_data_points = sum(category_counts)
    print(" ")
    print("===============")
    print("Data Distribution:")

    for i in range(total_categories):
        category_count = category_counts[i]
        category_percentage = float(category_count) / total_data_points
        print(f'Category{i} contains: {category_count}, {category_percentage: .3f}')

def split_data_by_category(attribute, category_counts):
    category_train_list = []
    category_test_list = []

    for x in range(len(category_counts)):
        if not category_counts[x] <5:
            category_data = df[df[attribute] == x]
            category_train, category_test = train_test_split(category_data, test_size=0.2)
            category_train_list.append(category_train)
            category_test_list.append(category_test)
    return category_train_list, category_test_list

def create_models(attribute, train, test):
    training_corpus = train['Answerq2q3Combined'].values
    training_labels = train[attribute].values
    test_corpus = test['Answerq2q3Combined'].values
    test_labels = test[attribute].values

    # Create TF-IDF Features
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(training_corpus)
    X = tfidf_vectorizer.fit_transform(training_corpus).todense()

    # Save fitted vectorizer
    filename = f'../../../CharGen/Server/models/male-base-models/male_{attribute}_X.sav'
    pickle.dump(X, open(filename, 'wb'))

    featurized_training_data = []
    for x in range(0, len(X)):
        tfidFeatures = np.array(X[x][0]).reshape(-1, )
        featurized_training_data.append(tfidFeatures)

    # Generate Feature Headers
    FeatureNames = []
    dimension = X.shape[1]

    for x in range(0, dimension):
        FeatureNames.append("TFIDF_" + str(x))

    X = tfidf_vectorizer.transform(test_corpus).todense()

    featurized_test_data = []
    for x in range(0, len(X)):
        tfidFeatures = np.array(X[x][0]).reshape(-1, )
        featurized_test_data.append(tfidFeatures)


    TargetNamesStrings = [str(i) for i in range(male_num_of_categories[attribute])]
    TargetNames = np.arange(male_num_of_categories[attribute])


    train = pd.DataFrame(featurized_training_data, columns=FeatureNames)
    test = pd.DataFrame(featurized_test_data, columns=FeatureNames)
    train['categories'] = pd.Categorical.from_codes(training_labels, TargetNames)
    test['categories'] = pd.Categorical.from_codes(test_labels, TargetNames)


    # Show the number of observations for the test and training dataframes
    print(" ")
    print("===============")
    print("Fold Information: ")
    print('Number of observations in the training data:', len(train))
    print('Number of features generated:', str(dimension))
    print(" ")
    print('Number of observations in the test data:', len(test))

    # Create a list of the feature column's names
    features = train.columns[:dimension]

    # Create a random forest classifier. By convention, clf means 'classifier'
    clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced")

    # Train the classifier to take the training features and learn how they relate to the training y (the stressors)
    clf.fit(train[features], train['categories'])

    # Apply the classifier we trained to the test data (which, remember, it has never seen before)
    preds = clf.predict(test[features])

    # View the PREDICTED stressors for the first five observations
    print(" ")
    print("===============")
    print("Example Prediction: ")
    print(preds[0:5])

    # View the ACTUAL stressors for the first five observations
    print(" ")
    print("===============")
    print("Actual: ")
    print(str(test['categories'].head()))

    # Create confusion matrix
    print(" ")
    print("===============")
    print("Confusion Matrix: ")
    print(" ")
    confusion_matrix = pd.crosstab(test['categories'], preds, rownames=['Actual Categories'],
                                   colnames=['Predicted Categories'])
    print(
        str(pd.crosstab(test['categories'], preds, rownames=['Actual Categories'], colnames=['Predicted Categories'])))

    # Show confusion matrix in a separate window
    sn.set(font_scale=1.4)  # for label size
    g = sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 12}, cmap="YlGnBu", cbar=False)  # font size
    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.5, top - 0.5)
    g.set_title("Male " + attribute)
    plt.tight_layout()
    plt.show()

    # Precision, Recall, F1 Scores rounded to 3 decimal places
    print(" ")
    print("Precision, Recall, Fbeta Stats: ")
    precision, recall, f1, _ = precision_recall_fscore_support(test['categories'], preds, average='macro')
    print('Macro:  ', f"Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1: {round(f1, 3)}")

    precision, recall, f1, _ = precision_recall_fscore_support(test['categories'], preds, average='micro')
    print('Micro:  ', f"Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1: {round(f1, 3)}")

    precision, recall, f1, _ = precision_recall_fscore_support(test['categories'], preds, average='weighted')
    print('Weighted', f"Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1: {round(f1, 3)}")
    index = min(test['categories'].unique()[-1], train['categories'].unique()[-1])

    #print(classification_report(test['categories'], preds, target_names=TargetNamesStrings[:index+1]))

    # View a list of the features and their importance scores
    print(" ")
    print("===============")
    print("Top Features: ")
    print(str(list(zip(train[features], clf.feature_importances_))))

    # save the model to disk
    filename = f'../../../CharGen/Server/models/male-base-models/male_{attribute}.sav'
    pickle.dump(clf, open(filename, 'wb'))

    filename = f'../../../CharGen/Server/models/male-base-models/male_{attribute}_vectorizer.sav'
    pickle.dump(tfidf_vectorizer, open(filename, 'wb'))





# Read data from converted/compiled CSV
df = pd.read_csv("male_data.csv")
male_num_of_categories = {"Hair": 17, "Hat": 4, "TShirt":15, "Pants": 15, "Shoes": 16, "Beard": 11, "Accessory": 14}

# Read each document and clean it.
df["Answerq2q3Combined"] = df["Answerq2q3Combined"].apply(clean)

# Preview the first 5 lines of the loaded data
print(df.head())

valid_male_attributes = ["Hair","Hat","TShirt","Pants","Shoes","Beard","Accessory"]
for attribute in valid_male_attributes:
    add_one_to_data(attribute)
    category_counts = count_num_of_data_per_category(attribute)
    CategoryLabels = list(df[attribute])
    print_data_distribution(category_counts)

    train_and_test_data = split_data_by_category(attribute, category_counts)
    print(category_counts)

    train = pd.concat(train_and_test_data[0], ignore_index=True)
    test = pd.concat(train_and_test_data[1], ignore_index=True)
    df['is_train'] = 0
    df.loc[train.index, 'is_train'] = 1
    create_models(attribute,train,test)

exit()


