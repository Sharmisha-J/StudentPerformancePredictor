import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
def load_data():
    df = pd.read_csv("data/student_data.csv")
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
    df['pass'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)
    return df


# ---------------- LEVEL 1 ----------------
def level1():
    print("\nLEVEL 1")

    df = load_data()
    X = df[['math score','reading score','writing score']]
    y = df['pass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

    print("Prediction:", "PASS" if model.predict([[70,75,80]])[0] else "FAIL")


# ---------------- LEVEL 2 ----------------
def level2():
    print("\nLEVEL 2")

    os.makedirs("outputs", exist_ok=True)

    df = load_data()

    sns.heatmap(df[['math score','reading score','writing score','pass']].corr(), annot=True)
    plt.savefig("outputs/heatmap.png")
    plt.show()

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
            best_model, best_acc = model, acc

        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
        plt.title(name)
        plt.show()

    joblib.dump(best_model, "model.pkl")
    print("Model Saved!")


# ---------------- LEVEL 3 ----------------
def level3():
    import streamlit as st

    model = joblib.load("model.pkl")

    st.title("Student Predictor")

    m = st.slider("Math", 0, 100, 60)
    r = st.slider("Reading", 0, 100, 60)
    w = st.slider("Writing", 0, 100, 60)

    if st.button("Predict"):
        pred = model.predict(np.array([[m, r, w]]))
        st.success("PASS" if pred[0] else "FAIL")


# ---------------- MENU ----------------
if __name__ == "__main__":
    print("\nChoose Level:")
    print("1 - Basic ML")
    print("2 - Advanced ML")
    print("3 - Streamlit App")

    ch = int(input("Enter choice: "))

    if ch == 1:
        level1()
    elif ch == 2:
        level2()
    elif ch == 3:
        print("Run with: streamlit run ml_all_in_one.py")
