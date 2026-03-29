import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Create target
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df['pass'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)

# Features
X = df[['math score', 'reading score', 'writing score']]
y = df['pass']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict
sample = [[70, 75, 80]]
prediction = model.predict(sample)

print("Prediction:", "PASS" if prediction[0] else "FAIL")
