# ==============================
# LEVEL 1: Basic ML Model
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'attendance': [50,55,60,65,70,75,80,85,90,95],
    'previous_score': [40,45,50,55,60,65,70,75,80,85],
    'pass': [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[['study_hours', 'attendance', 'previous_score']]
y = df['pass']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict new student
new_student = [[6, 80, 70]]
prediction = model.predict(new_student)

print("\nPrediction:")
print("PASS" if prediction[0] == 1 else "FAIL")
