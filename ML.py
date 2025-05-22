import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import loguniform
from sklearn.preprocessing import LabelEncoder, label_binarize, LabelBinarizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_excel('./Resampled_Data_pca_results.xlsx')
le = LabelEncoder()
y = le.fit_transform(df['group'])
X = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train_binary = label_binarize(y_train, classes=np.unique(y))
y_test_binary = label_binarize(y_test, classes=np.unique(y))

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Multilayer Perceptron': MLPClassifier(early_stopping=True, validation_fraction=0.2),
}

param_grids = {
    'Decision Tree': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 6),
        'max_features': [None, 'sqrt', 'log2']
    },
    'Gradient Boosting': {
        'loss': ['log_loss'],
        'learning_rate': loguniform(1e-3, 1e-1),
        'n_estimators': np.arange(50, 501, 50),
        'subsample': np.linspace(0.5, 0.9, 5),
        'max_features': ['sqrt', 'log2'],
        'criterion': ['friedman_mse'],
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 6),
        'max_depth': np.arange(3, 11),
        'min_impurity_decrease': np.linspace(0.0, 0.05, 6),
        'max_leaf_nodes': np.arange(10, 31, 5),
        'ccp_alpha': np.linspace(0.0, 0.05, 6)
    },
    'Support Vector Machine': {
        'C': loguniform(1e-4, 1e4),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': loguniform(1e-4, 1e4),
        'decision_function_shape': ['ovr', 'ovo']
    },
    'Random Forest': {
        'n_estimators': np.arange(100, 501, 50),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 6),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': np.arange(1, 31),
        'weights': ['uniform', 'distance'],
        'leaf_size': np.arange(10, 51),
        'p': [1, 2, 3],
        'metric': ['minkowski', 'manhattan', 'euclidean', 'chebyshev']
    },
    'Multilayer Perceptron': {
        'hidden_layer_sizes': [(20,), (30,), (10, 10), (15, 15), (5, 5, 5), (10, 10, 10)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': np.logspace(-4, 0, 5),
        'learning_rate_init': np.logspace(-4, -2, 3),
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
}

best_models = {}
final_params = {}
final_scores = {}
for name, model in models.items():
        param_grid = param_grids[name]
        rscv = RandomizedSearchCV(
            model, 
            param_distributions=param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='accuracy', 
            n_iter=20,
            random_state=42,
            error_score='raise',
            n_jobs=-1
        )
        rscv.fit(X_train, y_train)
        best_model = rscv.best_estimator_
        best_models[name] = best_model
        final_params[name] = rscv.best_params_
        final_scores[name] = rscv.best_score_

train_scores = {}
for name, model in best_models.items():
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    precision = precision_score(y_train, y_train_pred, average='weighted')
    recall = recall_score(y_train, y_train_pred, average='weighted')
    f1 = f1_score(y_train, y_train_pred, average='weighted')
    accuracy = accuracy_score(y_train, y_train_pred)
    roc_auc = roc_auc_score(y_train_binary, y_train_prob, multi_class='ovr', average='weighted')
    train_scores[name] = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'ROC AUC': roc_auc
    }

performance_metrics = {}
for name, model in best_models.items():
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr', average='weighted')
    performance_metrics[name] = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy':accuracy,
        'ROC AUC': roc_auc
    }

output_file_path = './roc_and_confusion_matrices.xlsx'   
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    lb = LabelBinarizer()
    y_train_binarized = lb.fit_transform(y_train)
    y_test_binarized = lb.transform(y_test)
    
    roc_data = {
        'Model': [],
        'Dataset': [],
        'Class': [],
        'FPR': [],
        'TPR': [],
        'AUC': []
    }
    
    for name, model in best_models.items():
        y_prob_train = model.predict_proba(X_train)
        for i in range(y_train_binarized.shape[1]):
            fpr_train, tpr_train, _ = roc_curve(y_train_binarized[:, i], y_prob_train[:, i])
            roc_auc_train = auc(fpr_train, tpr_train)
            for f, t in zip(fpr_train, tpr_train):
                roc_data['Model'].append(name)
                roc_data['Dataset'].append('Train')
                roc_data['Class'].append(i)
                roc_data['FPR'].append(f)
                roc_data['TPR'].append(t)
                roc_data['AUC'].append(roc_auc_train)
        
        y_prob_test = model.predict_proba(X_test)
        for i in range(y_test_binarized.shape[1]):
            fpr_test, tpr_test, _ = roc_curve(y_test_binarized[:, i], y_prob_test[:, i])
            roc_auc_test = auc(fpr_test, tpr_test)
            for f, t in zip(fpr_test, tpr_test):
                roc_data['Model'].append(name)
                roc_data['Dataset'].append('Test')
                roc_data['Class'].append(i)
                roc_data['FPR'].append(f)
                roc_data['TPR'].append(t)
                roc_data['AUC'].append(roc_auc_test)
    
    roc_df = pd.DataFrame(roc_data)
    roc_df.to_excel(writer, sheet_name='ROC Data', index=False)
    
    confusion_data = {
        'Model': [],
        'Dataset': [],
        'True Label': [],
        'Predicted Label': [],
        'Count': []
    }
    
    for name, model in best_models.items():
        cm_train = confusion_matrix(y_train, model.predict(X_train))
        for i in range(cm_train.shape[0]):
            for j in range(cm_train.shape[1]):
                confusion_data['Model'].append(name)
                confusion_data['Dataset'].append('Train')
                confusion_data['True Label'].append(i)
                confusion_data['Predicted Label'].append(j)
                confusion_data['Count'].append(cm_train[i, j])
        
        cm_test = confusion_matrix(y_test, model.predict(X_test))
        for i in range(cm_test.shape[0]):
            for j in range(cm_test.shape[1]):
                confusion_data['Model'].append(name)
                confusion_data['Dataset'].append('Test')
                confusion_data['True Label'].append(i)
                confusion_data['Predicted Label'].append(j)
                confusion_data['Count'].append(cm_test[i, j])
    
    confusion_df = pd.DataFrame(confusion_data)
    confusion_df.to_excel(writer, sheet_name='Confusion Matrix Data', index=False)
print(f"ROC和混淆矩阵数据已保存到 '{output_file_path}'")