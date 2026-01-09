import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"Makine Öğrenmesi\Proje Belgeleri\Kodlar\SVM\weather_classification_data.csv")

label_encoders = {}
categorical_columns = ['Cloud Cover', 'Season', 'Location', 'Weather Type']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=['Weather Type'])
y = data['Weather Type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

models = {
    "SVC": SVC(random_state=1),
    "LinearSVC": LinearSVC(random_state=1, max_iter=10000),
    "NuSVC": NuSVC(random_state=1)
}

for name, model in models.items():
    print(f"--- {name} ---")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print("Model Accuracy Score:", score)
    print("\nClassification Report:\n", report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoders['Weather Type'].classes_,
                yticklabels=label_encoders['Weather Type'].classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {name}')
    plt.show()
