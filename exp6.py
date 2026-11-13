#Experiment no 6 - Implementation of Classification Techniques for prediction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report


# load dataset
iris = load_iris()

df = pd.DataFrame(iris.data , columns=iris.feature_names)
df['Target'] = iris.target
print(df.sample(5))

# Separate features and target
x = df.drop(columns='Target')
y = df['Target']

# spliting data for train and test
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

print('\nTraining set size : ' , x_train.shape)
print('\nTesting set size : ' , x_test.shape)

#Decision Tree Model
model = DecisionTreeClassifier(random_state=42)

model.fit(x_train , y_train)

# model prediction
y_pred = model.predict(x_test)

# Accuracy Score
accuracy = accuracy_score(y_test , y_pred)

print('\nAccuracy Score: ' , accuracy)

# classification report
cr = classification_report(y_test , y_pred , target_names=iris.target_names)
print('\nClassification Report: \n' , cr)

# Confusion matrix

cm = confusion_matrix(y_test , y_pred)

print('\nConfusion matrix: \n' , cm)

# plotting heatmap to confusion matrix
plt.figure(figsize=(6, 4)) 
sns.heatmap(cm , annot=True , fmt='d' , xticklabels=iris.target_names , yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.show()

print('\nSample of Dataset: \n' , df.sample(7))

# create new flower data for prediction
new_flower = pd.DataFrame(
    [[6.5 , 2.5 , 5.5 , 2.2]],
    columns=iris.feature_names
)

print('\nNew Flower dataset for predicting the class: \n' , new_flower)
pred = model.predict(new_flower)
print('\nPrediction : ' , iris.target_names[pred[0]])