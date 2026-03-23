import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv(r"Makine Öğrenmesi\Proje Belgeleri\Kodlar\LogisticRegression\weather_classification_data.csv")

target_column = 'Weather Type' 

features = data.drop([target_column], axis=1)
target = data[target_column]

numeric_features = features.select_dtypes(include=[np.number])

x = (numeric_features - numeric_features.min()) / (numeric_features.max() - numeric_features.min())

if target.dtype == 'object':
    target = target.astype('category').cat.codes

x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(x_train, y_train)

score = log_reg.score(x_test, y_test)
print("Model Accuracy: ", score)

prediction = log_reg.predict(x_test)
cm = confusion_matrix(y_test, prediction)
print("Classification Report:\n", classification_report(y_test, prediction))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


