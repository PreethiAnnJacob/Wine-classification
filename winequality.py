'''Differentiate red and white wine based on the properties. TRYING https://www.datacamp.com/community/tutorials/deep-learning-python'''

'''________________Loading in the Data______________________'''

# Import pandas 
import pandas as pd

# Read in white wine data 
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

'''___________________Preprocess data__________________________'''

# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

# Isolate target labels
y = wines.quality

# Isolate data
X = wines.drop('quality', axis=1) 

from sklearn.model_selection import train_test_split

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

'''_________Scale the data____________'''

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Scale the data with `StandardScaler`
X = StandardScaler().fit_transform(X)


'''__________________Model neural network architecture________________'''

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the model
model = Sequential()

# Add input layer 
model.add(Dense(64, input_dim=12, activation='relu'))
    
# Add output layer 
model.add(Dense(1))

'''____________Compile and fit______________'''

import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, y):
    model = Sequential()
    model.add(Dense(64, input_dim=12, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(X[train], y[train], epochs=10, verbose=1)

'''_______Evaluate model_______________'''

score = model.evaluate(X_test, y_test,verbose=1)
mse_value, mae_value = model.evaluate(X[test], y[test], verbose=0)

print(mse_value)
print(mae_value)

'''______________________________'''

#from sklearn.metrics import r2_score

#r2_score(y[test], y_pred)

'''Epoch 1/10
5195/5195 [==============================] - 0s 72us/step - loss: 14.2576 - mean_absolute_error: 3.3457
Epoch 2/10
5195/5195 [==============================] - 0s 46us/step - loss: 1.6629 - mean_absolute_error: 1.0018
Epoch 3/10
5195/5195 [==============================] - 0s 31us/step - loss: 1.0362 - mean_absolute_error: 0.7853
Epoch 4/10
5195/5195 [==============================] - 0s 25us/step - loss: 0.7782 - mean_absolute_error: 0.6787
Epoch 5/10
5195/5195 [==============================] - 0s 24us/step - loss: 0.6443 - mean_absolute_error: 0.6187
Epoch 6/10
5195/5195 [==============================] - 0s 24us/step - loss: 0.5852 - mean_absolute_error: 0.5913
Epoch 7/10
5195/5195 [==============================] - 0s 26us/step - loss: 0.5511 - mean_absolute_error: 0.5772
Epoch 8/10
5195/5195 [==============================] - 0s 25us/step - loss: 0.5263 - mean_absolute_error: 0.5648
Epoch 9/10
5195/5195 [==============================] - 0s 29us/step - loss: 0.5129 - mean_absolute_error: 0.5582
Epoch 10/10
5195/5195 [==============================] - 0s 25us/step - loss: 0.5011 - mean_absolute_error: 0.5527
Epoch 1/10
5197/5197 [==============================] - 0s 53us/step - loss: 12.4715 - mean_absolute_error: 3.1039
Epoch 2/10
5197/5197 [==============================] - 0s 25us/step - loss: 1.5050 - mean_absolute_error: 0.9504
Epoch 3/10
5197/5197 [==============================] - 0s 28us/step - loss: 0.9353 - mean_absolute_error: 0.7493
Epoch 4/10
5197/5197 [==============================] - 0s 25us/step - loss: 0.7186 - mean_absolute_error: 0.6590
Epoch 5/10
5197/5197 [==============================] - 0s 28us/step - loss: 0.6197 - mean_absolute_error: 0.6112
Epoch 6/10
5197/5197 [==============================] - 0s 26us/step - loss: 0.5637 - mean_absolute_error: 0.5847
Epoch 7/10
5197/5197 [==============================] - 0s 24us/step - loss: 0.5377 - mean_absolute_error: 0.5698
Epoch 8/10
5197/5197 [==============================] - 0s 24us/step - loss: 0.5166 - mean_absolute_error: 0.5597
Epoch 9/10
5197/5197 [==============================] - 0s 25us/step - loss: 0.5021 - mean_absolute_error: 0.5498
Epoch 10/10
5197/5197 [==============================] - 0s 24us/step - loss: 0.4946 - mean_absolute_error: 0.5464
Epoch 1/10
5197/5197 [==============================] - 0s 58us/step - loss: 13.1266 - mean_absolute_error: 3.1861
Epoch 2/10
5197/5197 [==============================] - 0s 29us/step - loss: 1.5633 - mean_absolute_error: 0.9607
Epoch 3/10
5197/5197 [==============================] - 0s 27us/step - loss: 0.9597 - mean_absolute_error: 0.7517
Epoch 4/10
5197/5197 [==============================] - 0s 29us/step - loss: 0.7308 - mean_absolute_error: 0.6546
Epoch 5/10
5197/5197 [==============================] - 0s 24us/step - loss: 0.6163 - mean_absolute_error: 0.6046
Epoch 6/10
5197/5197 [==============================] - 0s 25us/step - loss: 0.5562 - mean_absolute_error: 0.5773
Epoch 7/10
5197/5197 [==============================] - 0s 24us/step - loss: 0.5257 - mean_absolute_error: 0.5620
Epoch 8/10
5197/5197 [==============================] - 0s 25us/step - loss: 0.5081 - mean_absolute_error: 0.5551
Epoch 9/10
5197/5197 [==============================] - 0s 24us/step - loss: 0.4961 - mean_absolute_error: 0.5491
Epoch 10/10
5197/5197 [==============================] - 0s 25us/step - loss: 0.4873 - mean_absolute_error: 0.5447
Epoch 1/10
5199/5199 [==============================] - 0s 58us/step - loss: 13.0713 - mean_absolute_error: 3.1693
Epoch 2/10
5199/5199 [==============================] - 0s 25us/step - loss: 1.5565 - mean_absolute_error: 0.9646
Epoch 3/10
5199/5199 [==============================] - 0s 25us/step - loss: 0.9817 - mean_absolute_error: 0.7641
Epoch 4/10
5199/5199 [==============================] - 0s 26us/step - loss: 0.7301 - mean_absolute_error: 0.6568
Epoch 5/10
5199/5199 [==============================] - 0s 24us/step - loss: 0.6210 - mean_absolute_error: 0.6029
Epoch 6/10
5199/5199 [==============================] - 0s 51us/step - loss: 0.5666 - mean_absolute_error: 0.5762
Epoch 7/10
5199/5199 [==============================] - 0s 26us/step - loss: 0.5361 - mean_absolute_error: 0.5625
Epoch 8/10
5199/5199 [==============================] - 0s 24us/step - loss: 0.5161 - mean_absolute_error: 0.5533
Epoch 9/10
5199/5199 [==============================] - 0s 26us/step - loss: 0.5029 - mean_absolute_error: 0.5465
Epoch 10/10
5199/5199 [==============================] - 0s 25us/step - loss: 0.4948 - mean_absolute_error: 0.5427
Epoch 1/10
5200/5200 [==============================] - 0s 70us/step - loss: 11.8302 - mean_absolute_error: 2.9679
Epoch 2/10
5200/5200 [==============================] - 0s 26us/step - loss: 1.6024 - mean_absolute_error: 0.9820
Epoch 3/10
5200/5200 [==============================] - 0s 26us/step - loss: 1.0070 - mean_absolute_error: 0.7746
Epoch 4/10
5200/5200 [==============================] - 0s 26us/step - loss: 0.7714 - mean_absolute_error: 0.6727
Epoch 5/10
5200/5200 [==============================] - 0s 26us/step - loss: 0.6512 - mean_absolute_error: 0.6167
Epoch 6/10
5200/5200 [==============================] - 0s 28us/step - loss: 0.5894 - mean_absolute_error: 0.5870
Epoch 7/10
5200/5200 [==============================] - 0s 25us/step - loss: 0.5517 - mean_absolute_error: 0.5723
Epoch 8/10
5200/5200 [==============================] - 0s 25us/step - loss: 0.5306 - mean_absolute_error: 0.5597
Epoch 9/10
5200/5200 [==============================] - 0s 24us/step - loss: 0.5133 - mean_absolute_error: 0.5539
Epoch 10/10
5200/5200 [==============================] - 0s 24us/step - loss: 0.5010 - mean_absolute_error: 0.5472
2145/2145 [==============================] - 0s 32us/step
0.48618812239804265
0.5480300569957095'''