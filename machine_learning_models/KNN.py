import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"Makine Öğrenmesi\\Proje Belgeleri\\Kodlar\\ANN\\weather_classification_data.csv")

print("Veri Sütunları:", data.columns)

y_col = 'Weather Type'  
if y_col not in data.columns:
    raise KeyError(f"Hedef sütun '{y_col}' veri çerçevesinde bulunamadı. Mevcut sütunlar: {list(data.columns)}")

X = data.drop(columns=[y_col])
y = data[y_col]

categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print("Kategorik Sütunlar:", list(categorical_columns))
    X[categorical_columns] = X[categorical_columns].apply(LabelEncoder().fit_transform)

if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

score_list = []
metric_list = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

for mtrc in metric_list:
    for each in range(1, 20, 2):
        knn = KNeighborsClassifier(n_neighbors=each, metric=mtrc)
        knn.fit(x_train, y_train)
        prediction = knn.predict(x_test)
        accuracy = knn.score(x_test, y_test)
        score_list.append(accuracy)

        cm = confusion_matrix(y_test, prediction)
        print("k={} Uzaklık Yöntemi={} Doğruluk Değeri: {:.2f}".format(each, mtrc, accuracy))
        print(classification_report(y_test, prediction))

        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f"Confusion Matrix (k={each}, Metric={mtrc})")
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek")
        plt.show()

    plt.plot(range(1, 20, 2), score_list, 'o-', color='blue', markerfacecolor='red', markersize=10)
    plt.title(f"Doğruluk Değerleri (Metric: {mtrc})")
    plt.xlabel("k Değerleri")
    plt.ylabel("Doğruluk")
    plt.show()

    score_list.clear()
