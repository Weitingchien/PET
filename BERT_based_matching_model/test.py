from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import make_scorer, f1_score

import numpy as np

def ordered_f1_score(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    precision = correct / len(y_pred) if y_pred else 0
    recall = correct / len(y_true) if y_true else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1



def main():


    all_predictions = ['fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear']
    all_labels = ['fear', 'fear', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear'] 
    print(ordered_f1_score(all_labels, all_predictions))

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'f1: {f1}')


    y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1 score: {macro_f1}")

if __name__ == "__main__":
    main()