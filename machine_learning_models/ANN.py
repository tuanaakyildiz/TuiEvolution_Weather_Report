import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv(r"Makine Öğrenmesi\Proje Belgeleri\Kodlar\ANN\weather_classification_data.csv")

weather_labels = data["Weather Type"].astype("category").cat.categories
data["Weather Type"] = data["Weather Type"].astype("category").cat.codes
data["Cloud Cover"] = data["Cloud Cover"].astype("category").cat.codes
data["Season"] = data["Season"].astype("category").cat.codes
data["Location"] = data["Location"].astype("category").cat.codes

y = data["Weather Type"].values
x_data = data.drop(["Weather Type"], axis=1)

x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

iterations = [500, 1000, 5000]

for max_iter in iterations:
    print(f"Results for max_iter = {max_iter}")
  
    mlp = MLPClassifier(max_iter=max_iter, random_state=1)
    mlp.fit(x_train, y_train)
    
    print("Score: ", mlp.score(x_test, y_test))
    prediction = mlp.predict(x_test)

    cm = confusion_matrix(y_test, prediction)
    print(classification_report(y_test, prediction, target_names=weather_labels))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=weather_labels, yticklabels=weather_labels)
    plt.title(f"Confusion Matrix (max_iter={max_iter})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

