"""# **3. Baseline Models (15%)**
work in progress

"""

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve, CalibrationDisplay


from main import tidied_training_set

data.iloc[1]

data = tidied_training_set.copy()

data = data.dropna(subset=['Distance_from_net'])

# Selecting the feature and the target
X = data[['Distance_from_net']]  # Features (2D array)
y = data['IsGoal']  # Target variable

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Creating a Logistic Regression classifier with default settings
clf = LogisticRegression()

# Training the classifier
clf.fit(X_train, y_train)

y_val_pred = clf.predict(X_val)

accuracy = (np.sum(y_val_pred==y_val))/len(y_val)

print(accuracy)

print(accuracy_score(y_val, y_val_pred))

print(np.bincount(y))
print(np.bincount(y_val_pred))

# Preprocessing the data
data = data.dropna(subset=['Distance_from_net', 'angle_from_net', 'IsGoal'])

# Features and target
X = data[['Distance_from_net', 'angle_from_net']]
y = data['IsGoal']

# Training and validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Function to train a model and evaluate it
def train_and_evaluate(feature_names):
    # Train the model
    clf = LogisticRegression()
    clf.fit(X_train[feature_names], y_train)

    # Predict probabilities
    y_proba = clf.predict_proba(X_val[feature_names])[:, 1]

    # Evaluate the model
    y_pred = clf.predict(X_val[feature_names])
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Accuracy for features {feature_names}: {accuracy}')
    plot_graphs(y_proba)


def plot_graphs(y_proba):
    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(1)
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # Plot goal rate vs. shot probability percentile
    plt.figure(2)
    percentiles = np.percentile(y_proba, np.arange(0, 101, 1))
    goal_rate = [np.mean(y_val[y_proba >= p]) for p in percentiles]
    plt.plot(np.arange(0, 101, 1), goal_rate)
    plt.xlabel('Shot Probability Model Percentile')
    plt.ylabel('Goal Rate')
    plt.title('Goal Rate vs. Shot Probability Model Percentile')

    # Plot cumulative goals vs. shot probability percentile
    plt.figure(3)
    cumulative_goals = [np.sum(y_val[y_proba >= p]) for p in percentiles]
    cumulative_goals = np.array(cumulative_goals) / np.sum(y_val)
    plt.plot(np.arange(0, 101, 1), cumulative_goals)
    plt.xlabel('Shot Probability Model Percentile')
    plt.ylabel('Cumulative Proportion of Goals')
    plt.title('Cumulative Proportion of Goals vs. Shot Probability Model Percentile')

    # Plot reliability diagram
    plt.figure(4)
    prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)
    CalibrationDisplay.from_predictions(y_val, y_proba, n_bins=10, strategy='uniform')
    plt.title('Reliability Diagram')

    # Show all plots
    plt.show()

    return y_proba


# Train and evaluate models on different features
y_proba_distance = train_and_evaluate(['Distance_from_net'])
y_proba_angle = train_and_evaluate(['angle_from_net'])
y_proba_both = train_and_evaluate(['Distance_from_net', 'angle_from_net'])
# Generate random baseline probabilities
random_probs = np.random.uniform(0, 1, size=len(y_val))
plot_graphs(random_probs)

