'''Differentiate red and white wine based on the properties. TRYING https://www.datacamp.com/community/tutorials/deep-learning-python'''

'''________________Loading in the Data______________________'''

# Import pandas 
import pandas as pd

# Read in white wine data 
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

#print(red)
#print(white)

'''___________________Data Exploration_____________________'''
'''
# Print info on white wine
print(white.info())

# Print info on red wine
print(red.info())
'''
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4898 entries, 0 to 4897
Data columns (total 12 columns):
fixed acidity           4898 non-null float64
volatile acidity        4898 non-null float64
citric acid             4898 non-null float64
residual sugar          4898 non-null float64
chlorides               4898 non-null float64
free sulfur dioxide     4898 non-null float64
total sulfur dioxide    4898 non-null float64
density                 4898 non-null float64
pH                      4898 non-null float64
sulphates               4898 non-null float64
alcohol                 4898 non-null float64
quality                 4898 non-null int64
dtypes: float64(11), int64(1)
memory usage: 459.3 KB
None
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1599 entries, 0 to 1598
Data columns (total 12 columns):
fixed acidity           1599 non-null float64
volatile acidity        1599 non-null float64
citric acid             1599 non-null float64
residual sugar          1599 non-null float64
chlorides               1599 non-null float64
free sulfur dioxide     1599 non-null float64
total sulfur dioxide    1599 non-null float64
density                 1599 non-null float64
pH                      1599 non-null float64
sulphates               1599 non-null float64
alcohol                 1599 non-null float64
quality                 1599 non-null int64
dtypes: float64(11), int64(1)
memory usage: 150.0 KB
None'''
'''
# First rows of `red` 
print("Head:\n",red.head())

# Last rows of `white`
print("Tail:\n",white.tail())

# Take a sample of 5 rows of `red`
print("Sample:\n",red.sample(5))

# Describe `white`
print("Describe white:\n",white.describe())

# Double check for null values in `red`
print("Double check for null in red:\n",pd.isnull(red))
'''

'''______________________Visualizing Data___________________________'''
'''
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
ax[0].legend(loc='best')
ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")

plt.show()
'''

'''____________________Verify visualised data using histogram_______________________________'''
'''
import numpy as np
print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
'''
'''(array([  0,   7, 673, 452, 305, 133,  21,   8], dtype=int64), array([ 7,  8,  9, 10, 11, 12, 13, 14, 15]))
(array([   0,  317, 1606, 1256,  906,  675,  131,    7], dtype=int64), array([ 7,  8,  9, 10, 11, 12, 13, 14, 15]))'''

''''--------------------------------------------------------------------------------'''
'''
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(red['quality'], red["sulphates"], color="red")
ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()
'''
'''________________Acidity__________________________'''
'''
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
redcolors = np.random.rand(6,4)
whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])
for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])
    
ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlim([0,1.7])
ax[1].set_xlim([0,1.7])
ax[0].set_ylim([5,15.5])
ax[1].set_ylim([5,15.5])
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("Alcohol") 
#ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
#fig.suptitle("Alcohol - Volatile Acidity")
fig.subplots_adjust(top=0.85, wspace=0.7)

plt.show()
'''

'''___________________Preprocess data__________________________'''

# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

'''____________Correlation matrix_________________'''
'''
import seaborn as sns
import matplotlib.pyplot as plt
corr = wines.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
'''

'''________Train and test data______________'''

# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
import numpy as np
# Specify the data 
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array
y= np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

'''____________Standardize the data__________________'''

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

'''__________________Model the data________________'''

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

'''_________See the model______'''

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

'''____________Compile and fit______________'''

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

'''________________Predict values______________'''

y_pred = model.predict(X_test)
y_pred[:5]
y_test[:5]

'''_______Evaluate model_______________'''

score = model.evaluate(X_test, y_test,verbose=1)

print(score)

'''______________________________'''

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

#print(y_test)
#print(y_pred)

# Confusion matrix
print(confusion_matrix(y_test.round(), y_pred.round()))

# Precision 
print(precision_score(y_test.round(), y_pred.round()))

# Recall
print(recall_score(y_test.round(), y_pred.round()))

# F1 score
print(f1_score(y_test.round(),y_pred.round()))

# Cohen's kappa
print(cohen_kappa_score(y_test.round(), y_pred.round()))

'''_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 12)                144
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 104
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 9
=================================================================
Total params: 257
Trainable params: 257
Non-trainable params: 0
_________________________________________________________________

Epoch 1/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0890 - acc: 0.9729
Epoch 2/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0247 - acc: 0.9947
Epoch 3/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0215 - acc: 0.9959
Epoch 4/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0194 - acc: 0.9956
Epoch 5/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0179 - acc: 0.9963
Epoch 6/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0160 - acc: 0.9970
Epoch 7/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0146 - acc: 0.9970
Epoch 8/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0145 - acc: 0.9963
Epoch 9/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0129 - acc: 0.9977
Epoch 10/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0146 - acc: 0.9966
Epoch 11/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0110 - acc: 0.9982
Epoch 12/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0113 - acc: 0.9977
Epoch 13/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0126 - acc: 0.9970
Epoch 14/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0102 - acc: 0.9979
Epoch 15/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0094 - acc: 0.9984
Epoch 16/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0127 - acc: 0.9982
Epoch 17/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0119 - acc: 0.9979
Epoch 18/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0122 - acc: 0.9977
Epoch 19/20
4352/4352 [==============================] - 5s 1ms/step - loss: 0.0123 - acc: 0.9979
Epoch 20/20
4352/4352 [==============================] - 4s 969us/step - loss: 0.0120 - acc: 0.9972
2145/2145 [==============================] - 0s 21us/step
[0.029450778697603038, 0.9948717948717949]
[[1585    3]
 [   8  549]]
0.9945652173913043
0.9856373429084381
0.9900811541929666
0.9866232169249897'''