import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

# Read data
@st.cache_data
def load_data():
    data = pd.read_csv("c:/688/heart.csv")
    return data

data = load_data()

# Create a Streamlit sidebar
st.sidebar.title("Heart Attack Prediction")

# Display dataset if user wants
if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Heart Attack Dataset")
    st.write(data)

# Split data
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

# Logistic Regression Model
X_train = train_data.drop(columns='output')
y_train = train_data['output']

X_test = test_data.drop(columns='output')
y_test = test_data['output']

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model Summary
st.subheader("Model Summary")
st.write("Coefficients:", model.coef_)
st.write("Intercept:", model.intercept_)

# Predictions
predictions = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

# ROC Curve
st.subheader("ROC Curve")
y_probs = cross_val_predict(model, X_train, y_train, cv=3, method="predict_proba")
fpr, tpr, thresholds = roc_curve(y_train, y_probs[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.grid(True)
st.pyplot(plt)

# Accuracy
st.subheader("Accuracy")
st.write("Accuracy:", accuracy)

# Prediction form
st.sidebar.subheader("Make Predictions")
age = st.sidebar.number_input("Enter Age", min_value=0, max_value=100, value=50)
sex = st.sidebar.selectbox("Select Gender", ['Male', 'Female'])
cp = st.sidebar.selectbox("Select Chest Pain Type", [0, 1, 2, 3])
trtbps = st.sidebar.number_input("Enter Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Enter Serum Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Select Fasting Blood Sugar Level", [0, 1])
restecg = st.sidebar.selectbox("Select Resting Electrocardiographic Results", [0, 1, 2])
thalachh = st.sidebar.number_input("Enter Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exng = st.sidebar.selectbox("Select Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("Enter ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=0.0)
slp = st.sidebar.selectbox("Select Slope of the Peak Exercise ST Segment", [0, 1, 2])
caa = st.sidebar.selectbox("Select Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thall = st.sidebar.selectbox("Select Thal", [0, 1, 2, 3])

# Convert Gender to Numeric
sex_mapping = {'Male': 0, 'Female': 1}
sex_numeric = sex_mapping[sex]

# Create a DataFrame with user input values
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_numeric],
    'cp': [cp],
    'trtbps': [trtbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalachh': [thalachh],
    'exng': [exng],
    'oldpeak': [oldpeak],
    'slp': [slp],
    'caa': [caa],
    'thall': [thall]
})

# Make prediction
prediction = model.predict(input_data)

# Display prediction result
st.sidebar.subheader("Prediction")
if prediction[0] == 0:
    st.sidebar.error("Risk of Heart Attack: Low")
else:
    st.sidebar.error("Risk of Heart Attack: High")
