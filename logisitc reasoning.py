# Imports libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# reads the Kaggle Heart Disease dataset
data = pd.read_csv('heart.csv')

# defines the features and target variable
X = data[['age', 'chol', 'trestbps']]  
y = data['target']  # target: Heart Disease 0 = No  1 = Yes

# splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# applies feature scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# trains model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# function to allow user input for predictions
def predict_heart_disease():
    # Takes user input for age, cholesterol, and blood pressure
    age = float(input("Enter Age: "))
    cholesterol = float(input("Enter Cholesterol Level: "))
    blood_pressure = float(input("Enter Blood Pressure: "))

    # Creates input data for prediction
    user_data = pd.DataFrame([[age, cholesterol, blood_pressure]], columns=['age', 'chol', 'trestbps'])
    
    # Apply the same scaling used in training
    user_data_scaled = scaler.transform(user_data)
    
    # Predict the heart disease outcome
    prediction = log_reg.predict(user_data_scaled)
    
    # outputs the prediction 
    if prediction == 1:
        print("The model predicts: Heart Disease (1)")
    else:
        print("The model predicts: No Heart Disease (0)")

    # evaluates the model performance using the entire dataset
    y_pred = log_reg.predict(X_test)

    # performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Prints performance metrics
    print("--- Logistic Regression Performance on Test Data ---")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print("\nConfusion Matrix:\n", cm)

    # plots the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Heart Disease", "Heart Disease"], yticklabels=["No Heart Disease", "Heart Disease"])
    plt.title("Confusion Matrix: Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

predict_heart_disease()
