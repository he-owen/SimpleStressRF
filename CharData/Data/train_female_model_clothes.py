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

def count_unique_rgb_values(df, r_column, g_column, b_column, attribute):
    # Create a new column with RGB tuples
    df[attribute+"_rgb"] = list(zip(df[r_column], df[g_column], df[b_column]))

    # Use nunique() to count the number of unique RGB tuples
    unique_rgb_count = df[attribute+"_rgb"].nunique()

    # Create a mapping from RGB tuples to integers
    tuple_to_int_mapping = dict(zip(df[attribute+"_rgb"].unique(), range(unique_rgb_count)))

    # Add a new column 'rgb_int' that stores the integer mapping
    df[attribute+"_rgb_category"] = df[attribute+"_rgb"].map(tuple_to_int_mapping)

    return df

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
    training_corpus = train['AnswerCombined'].values
    training_labels = train[attribute].values
    test_corpus = test['AnswerCombined'].values
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
    g.set_title(attribute)
    plt.show()

    # Precision, Recall, F1
    print(" ")
    print("Precision, Recall, Fbeta Stats: ")
    print('Macro:  ', precision_recall_fscore_support(test['categories'], preds, average='macro'))
    print('Micro:  ', precision_recall_fscore_support(test['categories'], preds, average='micro'))
    print('Weighted', precision_recall_fscore_support(test['categories'], preds, average='weighted'))
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
male_num_of_categories = {"SkinColor_rgb_category": 9, "EyeColor_rgb_category": 7, "HairColor_rgb_category": 12}

# Read each document and clean it.
df["AnswerCombined"] = df["AnswerCombined"].apply(clean)

# Preview the first 5 lines of the loaded data
print(df.head())

# Example usage:
# Assuming you have a DataFrame named 'your_df' with columns 'red', 'green', and 'blue' for RGB values


df = count_unique_rgb_values(df, 'SkinColor.r', 'SkinColor.g', 'SkinColor.b', "SkinColor")
df = count_unique_rgb_values(df, 'EyeColor.r', 'EyeColor.g', 'EyeColor.b', "EyeColor")
df = count_unique_rgb_values(df, 'HairColor.r', 'HairColor.g', 'HairColor.b', "HairColor")
valid_male_attributes = ["SkinColor_rgb_category","EyeColor_rgb_category","HairColor_rgb_category"]
for attribute in valid_male_attributes:
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
