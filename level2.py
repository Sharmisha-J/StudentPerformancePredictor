import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Create folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Target
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df['pass'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)

# ---------------- VISUALIZATION ----------------

sns.heatmap(df[['math score','reading score','writing score','pass']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.show()

sns.scatterplot(x='math score', y='reading score', hue='pass', data=df)
plt.title("Math vs Reading")
plt.savefig("outputs/scatter.png")
plt.show()

# ---------------- ML ----------------

X = df[['math score','reading score','writing score']]
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

best_model, best_acc = None, 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(name, "Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(name)
    plt.savefig(f"outputs/{name}.png")
    plt.show()

joblib.dump(best_model, "model.pkl")

print("Best Model Saved!")
