import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

t1 = time.time()

mnist_data = pd.read_csv(os.getcwd()+'/train.csv')
os.getcwd()

# Full data
X = mnist_data[mnist_data.columns[1:]].values
y = mnist_data.label.values

n_train = int(2*X.shape[0]/3)

# Trainning data
X_train = X[:n_train, :]
y_train = y[:n_train]

# Testing Data
X_test = X[n_train:, :]
y_test = y[n_train:]

def Display (n):
    # Displaying few numbers of the set
    for i in range (n):
        plt.subplot(n, 1, i+1)
        image = X[i]
        image = image.reshape(28, 28)
        plt.imshow(image)
        plt.text(5, 3, str(y[i]), bbox={'facecolor': 'white', 'pad': 10})
    plt.show()

# Test
t2 = time.time()
print('Step 1 execution time: ' + str(t2-t1) + 's')

# Cross Validation
random_forest = RandomForestClassifier(n_estimators = 100)
cv = cross_val_score(random_forest, X, y, cv = 3, verbose = True)
print('Result of Cross Validation on Random Forest: ' + str(cv.mean()))

# Test
t2 = time.time()
print('Step 2 execution time: ' + str(t2-t1) + 's')

# Trainning and test by the random forest method
random_forest.fit(X_train, y_train)
random_forest.score(X_test, y_test)

# Test
t2 = time.time()
print('Step 3 execution time: ' + str(t2-t1) + 's')

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_test, random_forest.predict(X_test)), index = range(0,10), columns = range(0,10))
print(cm)

# Test
t2 = time.time()
print('Step 4 execution time: ' + str(t2-t1) + 's')

if __name__ == '__main__':
#    print(mnist_data.head()) #Dataset display
    image = X[501]
    image = image.reshape(28, 28)
    plt.imshow(image)
    plt.show()