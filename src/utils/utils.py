import numpy as np
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score 

def evaluate_model(model, X_val, y_val):
    # predict probabilities for ROC AUC
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    roc_auc = roc_auc_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    return {'ROC AUC': roc_auc, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
