import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\user\Desktop\Makine Öğrenmesi son\Makine Öğrenmesi\Proje Belgeleri\Kodlar\ANN\weather_classification_data.csv")

label_encoders = {}
categorical_columns = ["Cloud Cover", "Season", "Location", "Weather Type"]

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

y = data["Weather Type"]
X = data.drop(["Weather Type"], axis=1)

X = (X - X.min()) / (X.max() - X.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

mlp = MLPClassifier(max_iter=1000, random_state=1)
mlp.fit(X_train, y_train)

score = mlp.score(X_test, y_test)
predictions = mlp.predict(X_test)
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, target_names=label_encoders["Weather Type"].classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoders["Weather Type"].classes_, 
            yticklabels=label_encoders["Weather Type"].classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Score: ", score)
print(report)
