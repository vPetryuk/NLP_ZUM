import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Read preprocessed dataset
data = pd.read_csv('../csv_files/processed_data/preprocessed_data.tsv', sep='\t')
data['avg_vector'] = data['avg_vector'].apply(lambda x: ast.literal_eval(x))

# Divide data into train and test datasets
features = np.vstack(data['avg_vector'].values)
labels = data['cluster']
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the classifiers to be used
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier()
}

# Fit and evaluate each classifier
for classifier_name, classifier in classifiers.items():
    # Fit the classifier
    classifier.fit(features_train, labels_train)

    # Predict
    predictions = classifier.predict(features_test)

    # Performance evaluation
    confusion_mat = confusion_matrix(labels_test, predictions)
    roc_auc = roc_auc_score(label_binarize(labels_test, classes=[0, 1, 2]), label_binarize(predictions, classes=[0, 1, 2]),
                            average='macro')
    false_positive_rate, true_positive_rate, _ = roc_curve(label_binarize(labels_test, classes=[0, 1, 2]).ravel(),
                            label_binarize(predictions, classes=[0, 1, 2]).ravel())

    # Output results
    print(f"{classifier_name}\nConfusion Matrix:\n{confusion_mat}\nROC AUC Score: {roc_auc}")

    # Draw ROC curve
    plt.plot(false_positive_rate, true_positive_rate, label=f"{classifier_name} (AUC = {roc_auc:.2f})")

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
plt.title('ROC Curve Analysis', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.show()

