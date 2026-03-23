import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"Makine Öğrenmesi\Proje Belgeleri\Kodlar\ANN\weather_classification_data.csv")

label_encoders = {}
categorical_columns = ['Cloud Cover', 'Season', 'Weather Type', 'Location']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=['Weather Type'])
y = data['Weather Type']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
prediction_gnb = gnb.predict(X_test)
gnb_score = gnb.score(X_test, y_test)
gnb_report = classification_report(y_test, prediction_gnb)
cm_gnb = confusion_matrix(y_test, prediction_gnb)

sns.heatmap(cm_gnb, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title('GaussianNB Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
prediction_mnb = mnb.predict(X_test)
mnb_score = mnb.score(X_test, y_test)
mnb_report = classification_report(y_test, prediction_mnb)
cm_mnb = confusion_matrix(y_test, prediction_mnb)

sns.heatmap(cm_mnb, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title('MultinomialNB Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
prediction_bnb = bnb.predict(X_test)
bnb_score = bnb.score(X_test, y_test)
bnb_report = classification_report(y_test, prediction_bnb)
cm_bnb = confusion_matrix(y_test, prediction_bnb)

sns.heatmap(cm_bnb, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title('BernoulliNB Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

results = {
    "GaussianNB": {"Score": gnb_score, "Classification Report": gnb_report},
    "MultinomialNB": {"Score": mnb_score, "Classification Report": mnb_report},
    "BernoulliNB": {"Score": bnb_score, "Classification Report": bnb_report},
}

for model, result in results.items():
    print(f"{model}:\nScore: {result['Score']}\nClassification Report:\n{result['Classification Report']}\n")
