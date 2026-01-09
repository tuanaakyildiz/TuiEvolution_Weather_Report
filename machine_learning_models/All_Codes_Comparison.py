import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

data = pd.read_csv(r"Makine Öğrenmesi\Proje Belgeleri\Kodlar\weather_classification_data.csv")

categorical_columns = ['Cloud Cover', 'Season', 'Location', 'Weather Type']
label_encoders = {}
for col in categorical_columns:
    le = pd.factorize(data[col])
    data[col] = le[0]
    label_encoders[col] = le[1]

X = data.drop(columns=['Weather Type'])
y = data['Weather Type']

X = (X - X.min()) / (X.max() - X.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Trees": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Artificial Neural Networks": MLPClassifier(max_iter=1000, random_state=42),
    "Support Vector Machines": SVC(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = f1

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='blue') 
plt.ylabel("F1 Score")
plt.ylim(0.85, 1.0)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

results
