import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# STEP 1: LOAD DATA
# ----------------------------
data = pd.read_csv("Qus.csv")

# Clean data (remove invalid rows)
data = data.dropna()
data = data[data['result'].between(0,9)]

# ----------------------------
# STEP 2: CREATE FEATURES
# ----------------------------
# Using previous 3 results to predict next

data['prev1'] = data['result'].shift(1)
data['prev2'] = data['result'].shift(2)
data['prev3'] = data['result'].shift(3)

data = data.dropna()

X = data[['prev1','prev2','prev3']]
y = data['result']

# ----------------------------
# STEP 3: SPLIT DATA
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# STEP 4: TRAIN MODEL (100 times strong)
# ----------------------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ----------------------------
# STEP 5: SELF TEST 50 RANDOM
# ----------------------------
correct = 0

for i in range(50):
    idx = random.randint(0, len(X_test)-1)
    sample = X_test.iloc[idx:idx+1]
    prediction = model.predict(sample)
    actual = y_test.iloc[idx]

    if prediction[0] == actual:
        correct += 1

accuracy = correct / 50 * 100

print("Self Test Accuracy:", accuracy, "%")

# ----------------------------
# STEP 6: FINAL ACCEPT CHECK
# ----------------------------
if accuracy >= 95:
    print("MODEL STATUS: ACCEPTED ✅")
else:
    print("MODEL STATUS: REJECTED ❌")

# ----------------------------
# STEP 7: NEXT PREDICTION
# ----------------------------
print("\nEnter last 3 results:")
p1 = int(input("Last 1: "))
p2 = int(input("Last 2: "))
p3 = int(input("Last 3: "))

next_pred = model.predict([[p1,p2,p3]])[0]

# Convert to Small / Big
if next_pred <= 4:
    size = "Small"
else:
    size = "Big"

print("\nNext Prediction:", next_pred, size)