# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as NumPy, Pandas, Matplotlib, and Scikit-learn.
2. Create a dataset with Hours Studied (X) and Marks Scored (y) and convert it into a DataFrame.
3. Split the dataset into training and testing sets using train_test_split().
4. Create and train the Linear Regression model using the training data.
5. Predict the marks for test data and calculate performance metrics like MSE and R² score.
6. Plot the regression line along with actual data points.
7. Predict marks for a new input (e.g., number of study hours) and display the result.
8. Stop the program. 

## Program:
# Program to implement the simple linear regression model for predicting the marks scored.
# Developed by: VIJIYALAKSHMI A
# RegisterNumber: 212225240185
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)


print("Dataset:\n", df.head())
df
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")

```
## Output:
<img width="704" height="344" alt="Screenshot 2026-04-27 140428" src="https://github.com/user-attachments/assets/8b4cd209-5923-40ad-a70e-eb60579de813" />
<img width="955" height="691" alt="Screenshot 2026-04-27 140439" src="https://github.com/user-attachments/assets/494886fb-cb8c-4a09-bb71-771f4e29c564" />
<img width="1706" height="82" alt="Screenshot 2026-04-27 140454" src="https://github.com/user-attachments/assets/d41acdf2-30f3-4a0b-88a7-c660b9379fe3" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
