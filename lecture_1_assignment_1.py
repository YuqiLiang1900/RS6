from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the data
digits = load_digits()
# Create the matrix of features 
X = digits.data
# Create the vector of targets
y = digits.target

# Feature exploration
print('There are {} rows and {} columns in the matrix of features'.format(X.shape[0], X.shape[1]))

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))

# Split the datset, 75% for training, and 25% for testing
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=7)

# Standarlization using Z-Score
standard_scaler = preprocessing.StandardScaler()
train_ss_X = standard_scaler.fit_transform(train_X)
test_ss_X = standard_scaler.transform(test_X)

# Create CART classifier
clf = DecisionTreeClassifier(criterion='gini')
# Fit the model
clf = clf.fit(train_ss_X, train_y)
# Predict
predict_y = clf.predict(test_ss_X)

# Model accuracy as the evaluation, i.e., how often is the classifier correct
print('Accuracy:', accuracy_score(test_y, predict_y))




