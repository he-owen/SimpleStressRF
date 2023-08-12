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


# Read data from converted/compiled CSV
df = pd.read_csv("male_data.csv")


# Preview the first 5 lines of the loaded data
print(df.head())

# Data Values: 0 1 2 3 4 5 6 7 8 9 10 11
# Convert label column to numeric labels
df.loc[df.HairColor.b == "0", 'HairColor.b'] = "0"
df.loc[df.HairColor.b == "1", 'HairColor.b'] = "1"
df.loc[df.HairColor.b == "2", 'HairColor.b'] = "2"
df.loc[df.HairColor.b == "3", 'HairColor.b'] = "3"
df.loc[df.HairColor.b == "4", 'HairColor.b'] = "4"
df.loc[df.HairColor.b == "5", 'HairColor.b'] = "5"
df.loc[df.HairColor.b == "6", 'HairColor.b'] = "6"
df.loc[df.HairColor.b == "7", 'HairColor.b'] = "7"
df.loc[df.HairColor.b == "8", 'HairColor.b'] = "8"
df.loc[df.HairColor.b == "9", 'HairColor.b'] = "9"
df.loc[df.HairColor.b == "10", 'HairColor.b'] = "10"
df.loc[df.HairColor.b == "11", 'HairColor.b'] = "11"

df["HairColor.b"] = df["HairColor.b"].astype(int)

# Read each document and clean it.
df["AnswerCombined"] = df["AnswerCombined"].apply(clean)


# Lets do some quick analysis
CategoryLabels = list(df["HairColor.b"])
Category0 = CategoryLabels.count(0)
Category1 = CategoryLabels.count(1)
Category2 = CategoryLabels.count(2)
Category3 = CategoryLabels.count(3)
Category4 = CategoryLabels.count(4)
Category5 = CategoryLabels.count(5)
Category6 = CategoryLabels.count(6)
Category7 = CategoryLabels.count(7)
Category8 = CategoryLabels.count(8)
Category9 = CategoryLabels.count(9)
Category10 = CategoryLabels.count(10)
Category11 = CategoryLabels.count(11)


print(" ")
print("===============")
print("Data Distribution:")
print('Category0 contains:', Category0, float(Category0) / float(len(CategoryLabels)))
print('Category1 contains:', Category1, float(Category1) / float(len(CategoryLabels)))
print('Category2 contains:', Category2, float(Category2) / float(len(CategoryLabels)))
print('Category3 contains:', Category3, float(Category3) / float(len(CategoryLabels)))
print('Category4 contains:', Category4, float(Category4) / float(len(CategoryLabels)))
print('Category5 contains:', Category5, float(Category5) / float(len(CategoryLabels)))
print('Category6 contains:', Category6, float(Category6) / float(len(CategoryLabels)))
print('Category0 contains:', Category7, float(Category7) / float(len(CategoryLabels)))
print('Category0 contains:', Category8, float(Category8) / float(len(CategoryLabels)))
print('Category0 contains:', Category9, float(Category9) / float(len(CategoryLabels)))
print('Category0 contains:', Category10, float(Category10) / float(len(CategoryLabels)))
print('Category0 contains:', Category11, float(Category11) / float(len(CategoryLabels)))


Category0_data = df[df['HairColor.b'] == 0]
Category1_data = df[df['HairColor.b'] == 1]
Category2_data = df[df['HairColor.b'] == 2]
Category3_data = df[df['HairColor.b'] == 3]
Category4_data = df[df['HairColor.b'] == 4]
Category5_data = df[df['HairColor.b'] == 5]
Category6_data = df[df['HairColor.b'] == 6]
Category7_data = df[df['HairColor.b'] == 7]
Category8_data = df[df['HairColor.b'] == 8]
Category9_data = df[df['HairColor.b'] == 9]
Category10_data = df[df['HairColor.b'] == 10]
Category11_data = df[df['HairColor.b'] == 11]


Category0_train, Category0_test = train_test_split(Category0_data, test_size=0.2)
Category1_train, Category1_test = train_test_split(Category1_data, test_size=0.2)
Category2_train, Category2_test = train_test_split(Category2_data, test_size=0.2)
Category3_train, Category3_test = train_test_split(Category3_data, test_size=0.2)
Category4_train, Category4_test = train_test_split(Category4_data, test_size=0.2)
Category5_train, Category5_test = train_test_split(Category5_data, test_size=0.2)
Category6_train, Category6_test = train_test_split(Category6_data, test_size=0.2)
Category7_train, Category0_test = train_test_split(Category7_data, test_size=0.2)
Category8_train, Category0_test = train_test_split(Category8_data, test_size=0.2)
Category9_train, Category0_test = train_test_split(Category9_data, test_size=0.2)
Category10_train, Category0_test = train_test_split(Category10_data, test_size=0.2)
Category11_train, Category0_test = train_test_split(Category11_data, test_size=0.2)


train = pd.concat([Category0_train, Category1_train, Category2_train, Category3_train, Category4_train, Category5_train, Category6_train, Category7_train, Category8_train, Category9_train, Category10_train, Category11_train, Category12_train, Category13_train, Category14_train])
test = pd.concat([Category0_test, Category1_test, Category2_test, Category3_test, Category4_test, Category5_test, Category6_test, Category7_test, Category8_test, Category9_test, Category10_test, Category11_test, Category12_test, Category13_test, Category14_test])

df['is_train'] = 0
df.loc[train.index, 'is_train'] = 1

training_corpus = train['AnswerCombined'].values
training_labels = train['HairColor.b'].values + 1
test_corpus = test['AnswerCombined'].values
test_labels = test['HairColor.b'].values + 1


# Create TF-IDF Features
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(training_corpus)
X = tfidf_vectorizer.fit_transform(training_corpus).todense()

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


# Create final dataframes
TargetNamesStrings = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
TargetNames = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

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
confusion_matrix = pd.crosstab(test['categories'], preds, rownames=['Actual Categories'], colnames=['Predicted Categories'])
print(str(pd.crosstab(test['categories'], preds, rownames=['Actual Categories'], colnames=['Predicted Categories'])))


# Show confusion matrix in a separate window
sn.set(font_scale=1.4)#for label size
g = sn.heatmap(confusion_matrix, annot=True,annot_kws={"size": 12}, cmap="YlGnBu", cbar=False)# font size
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# Precision, Recall, F1
print(" ")
print("Precision, Recall, Fbeta Stats: ")
print('Macro:  ', precision_recall_fscore_support(test['categories'], preds, average='macro'))
print('Micro:  ', precision_recall_fscore_support(test['categories'], preds, average='micro'))
print('Weighted', precision_recall_fscore_support(test['categories'], preds, average='weighted'))
print(classification_report(test['categories'], preds, target_names=TargetNamesStrings))


# View a list of the features and their importance scores
print(" ")
print("===============")
print("Top Features: ")
print(str(list(zip(train[features], clf.feature_importances_))))


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
exit()