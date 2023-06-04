# What Has Been Done
## General Overview
All attributes with integer data values for the male and female datasets have been run through updated versions of the train_model.py code. 

Each attribute has its own python file (e.g. `train_hat_male.py`), and running that file gives you information about things such as the data distribution (e.g. how many of the male characters in the male dataset have brown hair vs black hair) and the confusion matrix.

The summary of the outputs of all of those files is here: https://docs.google.com/presentation/d/1SE-Cpz0k7bA9SDmG8Z1wh8nLNrfc55XbMh1QAfJCDBk/edit?usp=sharing

## Updated train_model.py Code
The code in each of the attribute training files is essentially the same, with the main difference being the attribute examined and the number of categories.

The number of categories corresponds with the number of unique data values for the attribute in the male/female dataset.

### Changes Made to Original train_model.py Code
- Most of the attributes' unique data values include -1, so to avoid NaN issues, we added 1 to all training and testing labels for those attributes.
- After the large section of code that prints the data distribution in each of the attribute training files, there is new code that splits each of the individual categories into an 80-20 split for training and testing, respectively.
  - The original train_model.py code split all the data as a whole into an 80-20 split for training and testing, which was not ideal.

# Future Work Needed to Be Done
- A lot of the attributes have data values that skip a number (e.g. 0, 1, 3 => 2 is missing), which causes a ValueError in the code, which says that the number of classes does not match the size of `target_names`. A possible solution for this is to subtract 1 from data values after the gap to fill in that gap.
  - This issue is described in detail for each of the attributes that have it in the linked Slides above.
- Some attributes have a category that only has 1 piece of data (e.g. Male Shoes), which causes an issue in the code since that means one of the training or testing data is empty for that category. Adding more data to the datasets would probably be the best way to resolve this issue.
- The section of code in each of the attribute training files for splitting the training and testing data into an 80-20 split is very long and repetitive, so there is probably a better, more refined way to do that.