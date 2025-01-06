import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
file_path = 'C:/WASTE MANAGEMENT SYSTEM/waste_sensor_data.csv'
data = pd.read_csv(file_path)

# Step 1: Handle Missing Values
if data.isnull().sum().sum() > 0:
    print("Handling missing values...")
    # Fill missing numerical values with the median
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    
    # Fill missing categorical values with the mode
    cat_cols = data.select_dtypes(include=[object]).columns
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

# Step 2: Remove Duplicates
if data.duplicated().sum() > 0:
    print("Removing duplicate rows...")
    data = data.drop_duplicates()

# Step 3: Remove Outliers
print("Removing outliers...")
num_cols = data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Step 4: Drop Irrelevant Columns
irrelevant_cols = ['timestamp', 'sensor_id']  # Example irrelevant columns
data = data.drop(columns=[col for col in irrelevant_cols if col in data.columns])

# Step 5: Check for Imbalanced Dataset
class_distribution = data['waste_type'].value_counts()
print("Class distribution before balancing:")
print(class_distribution)

# Balance the dataset
majority_class = data[data['waste_type'] == class_distribution.idxmax()]
balanced_data = majority_class.copy()

for waste_type, count in class_distribution.items():
    if waste_type != class_distribution.idxmax():
        minority_class = data[data['waste_type'] == waste_type]
        resampled_minority = resample(minority_class, 
                                      replace=True, 
                                      n_samples=class_distribution.max(), 
                                      random_state=42)
        balanced_data = pd.concat([balanced_data, resampled_minority])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Feature Engineering
balanced_data['inductive_to_capacitive'] = balanced_data['inductive_property'] / (balanced_data['capacitive_property'] + 1e-9)
balanced_data['infrared_to_moisture'] = balanced_data['infrared_property'] / (balanced_data['moisture_property'] + 1e-9)

# Encode target variable ('waste_type')
label_encoder = LabelEncoder()
balanced_data['waste_type_encoded'] = label_encoder.fit_transform(balanced_data['waste_type'])

# Feature Selection
features = [
    'inductive_property', 'capacitive_property', 'moisture_property', 
    'infrared_property', 'inductive_to_capacitive', 'infrared_to_moisture'
]
X = balanced_data[features]
y = balanced_data['waste_type_encoded']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction with PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Apply Boosting to Random Forest

base_rf = RandomForestClassifier(random_state=24, n_estimators=25, max_depth=7)
boosted_rf = AdaBoostClassifier(estimator=base_rf, n_estimators=48, random_state=42)
# Fit the boosted model
boosted_rf.fit(X_train, y_train)

# Evaluate the model on training and testing sets
y_train_pred = boosted_rf.predict(X_train)
y_test_pred = boosted_rf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Display results
print(f"Boosted Random Forest Training Accuracy: {train_accuracy:.2f}")
print(f"Boosted Random Forest Testing Accuracy: {test_accuracy:.2f}")

# Additional visualizations and evaluations can be added as before
print("\nTraining Set Classification Report:")
print(classification_report(y_train, y_train_pred, target_names=label_encoder.classes_))

print("\nTesting Set Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# Plot Feature Importances for Boosted Random Forest
feature_importances = boosted_rf.feature_importances_
pca_feature_names = [f'PCA Component {i+1}' for i in range(len(feature_importances))]

sns.barplot(x=pca_feature_names, y=feature_importances)
plt.title("Boosted Random Forest Feature Importances (PCA Components)")
plt.xlabel("PCA Components")
plt.ylabel("Importance")
plt.show()

# PCA Explained Variance Ratio
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
sns.barplot(x=[f'PCA Component {i+1}' for i in range(len(explained_variance_ratio))], 
            y=explained_variance_ratio)
plt.title("PCA Explained Variance Ratio")
plt.xlabel("PCA Components")
plt.ylabel("Variance Ratio")
plt.show()

# Confusion Matrix for Testing Predictions
conf_matrix = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                       display_labels=label_encoder.classes_).plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix - Test Set")
plt.show()

# Correlation Heatmap for Sensor Properties (Example)
correlation_matrix = balanced_data[features].corr()

# Display the Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Sensor Properties")
plt.xlabel("Sensor Properties")
plt.ylabel("Sensor Properties")
plt.show()

# Final plots and display of confusion matrix and heatmap are now included in the script.
# Example sensor data for the 'organic' class
organic_data = pd.DataFrame({
    'inductive_property': [1],
    'capacitive_property': [0.21],
    'moisture_property': [0.51],
    'infrared_property': [91.0]
})

# Feature Engineering
organic_data['inductive_to_capacitive'] = organic_data['inductive_property'] / (organic_data['capacitive_property'] + 1e-9)
organic_data['infrared_to_moisture'] = organic_data['infrared_property'] / (organic_data['moisture_property'] + 1e-9)

# Select features
organic_X = organic_data[features]

# Feature Scaling
organic_X_scaled = scaler.transform(organic_X)

# Dimensionality Reduction with PCA
organic_X_pca = pca.transform(organic_X_scaled)

# Predict the class of new data
organic_predictions = boosted_rf.predict(organic_X_pca)

# Decode the predicted class labels
organic_predictions_labels = label_encoder.inverse_transform(organic_predictions)

# Output the classification result
print(f"Sensor data is classified as: {organic_predictions_labels[0]}")


# New sensor data
new_data = pd.DataFrame({
    'inductive_property': [0.32],
    'capacitive_property': [1.2],
    'moisture_property': [0.2],
    'infrared_property': [0.3]
})

# Feature Engineering
new_data['inductive_to_capacitive'] = new_data['inductive_property'] / (new_data['capacitive_property'] + 1e-9)
new_data['infrared_to_moisture'] = new_data['infrared_property'] / (new_data['moisture_property'] + 1e-9)

# Select features
new_X = new_data[features]

# Feature Scaling
new_X_scaled = scaler.transform(new_X)

# Dimensionality Reduction with PCA
new_X_pca = pca.transform(new_X_scaled)

# Predict the class of new data
new_predictions = boosted_rf.predict(new_X_pca)

# Decode the predicted class labels
new_predictions_labels = label_encoder.inverse_transform(new_predictions)

# Output the classification result
print(f"Sensor data is classified as: {new_predictions_labels[0]}")