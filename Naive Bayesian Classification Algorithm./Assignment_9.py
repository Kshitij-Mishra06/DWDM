import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from CSV file
# Make sure to replace 'path/to/your_file.csv' with the actual path to your CSV file.
file_path = 'I:/My Drive/DWDM/assignments/Assignment_9/Flipkart_Mobiles.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Preprocessing: Convert categorical variables to numeric using Label Encoding
label_encoder = LabelEncoder()
df['Brand'] = label_encoder.fit_transform(df['Brand'])
df['Model'] = label_encoder.fit_transform(df['Model'])
df['Color'] = label_encoder.fit_transform(df['Color'])
df['Memory'] = label_encoder.fit_transform(df['Memory'])
df['Storage'] = label_encoder.fit_transform(df['Storage'])

# Create a binary target variable for Rating (e.g., 1 if Rating >= 4.4, else 0)
df['Target'] = (df['Rating'] >= 4.4).astype(int)

# Features and target variable
X = df.drop(columns=['Rating', 'Selling Price', 'Original Price', 'Target'])
y = df['Target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)
