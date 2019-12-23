# SimpleStressRF
A simple scikit-learn, RF classifier that is based mTurk data collected for the Popbots project

# Dependencies
Code has been thorougly tested in PyCharm 2019.2.5 (Professional) and appears to work in IDLE, both using Python 3.7. Required libraries include: pickle for saving the model, pandas and numpy for managing the dataframes, String for manuiplating string datatype (i.e., input data), NLTK and Sklearn for some basic NLP and Machine Learning features, matplotlib and seaborn for plotting data.

# Operation
Category labels needs to be edited on ~Line 38ish - 79ish and looked for a TargetNamesStrings and TargetNames variables around ~Line 120 if you edit the input data. The script currently looks for a file called data.csv, which is on box. Once the category labels are updated, the code can be "run" in either of the editors mentioned above and a classification report will be generated; this might be useful if collecting and validating data from mTurk.
