import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import joblib

def split_data(train_path, test_path):
    train_pca_data = np.load(train_path)
    np.random.shuffle(train_pca_data)
    x_train = train_pca_data[:, :-1]
    y_train = train_pca_data[:,-1]

    test_pca_data = np.load(test_path)
    np.random.shuffle(test_pca_data)
    x_test = test_pca_data[:, :-1]
    y_test = test_pca_data[:,-1]
    
#     y_labels = pca_data[:, -1]
#     y_labels[y_labels != 0] = 1

#     x = pca_data[:, 0:-1]

#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y_labels, test_size=0.2, random_state=614, shuffle=True, stratify=y_labels)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    train_data_path = os.path.join(os.path.join("..", ".."), "data/pca_transformed_data/pca_data_400_400_100_train.npy")
    test_data_path = os.path.join(os.path.join("..", ".."), "data/pca_transformed_data/pca_data_400_400_100_test.npy")

    x_train, x_test, y_train, y_test = split_data(train_data_path, test_data_path)
    # SVM grid search CV
    svm_parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1.0]}

    svc = SVC(gamma='auto').fit(x_train, y_train)

    svm_cv = GridSearchCV(svc, svm_parameters, n_jobs=-1, return_train_score=True).fit(x_train, y_train)

    print("SVM accuracy score :- ",svm_cv.best_score_)

    #RF Grid Search CV

    rf_clf = RandomForestClassifier(random_state=614).fit(x_train, y_train)

    rf_parameters = {'n_estimators': [50, 100, 200], 'max_depth': [8, 16, 24, 50]}

    gscv_rfc = GridSearchCV(rf_clf, rf_parameters).fit(x_train, y_train)

    print("Random Forest score :-",gscv_rfc.best_score_)

    # Logistic regression CV

    clf_lg = LogisticRegression(random_state=614, max_iter=1000)
    scores = cross_val_score(clf_lg, x_train, y_train, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    print("SVM accuracy scores : - ")

    print(confusion_matrix(svm_cv.predict(x_test), y_test))
    print(precision_score(svm_cv.predict(x_test), y_test), recall_score(svm_cv.predict(x_test), y_test),
          accuracy_score(svm_cv.predict(x_test), y_test))

    print("RF accuracy scores : - ")

    print(confusion_matrix(gscv_rfc.predict(x_test), y_test))
    print(precision_score(gscv_rfc.predict(x_test), y_test), recall_score(gscv_rfc.predict(x_test), y_test),
          accuracy_score(gscv_rfc.predict(x_test), y_test))

    print("Logistic accuracy scores :- ")

    print(confusion_matrix(clf_lg.fit(x_train, y_train).predict(x_test), y_test))
    print(precision_score(clf_lg.fit(x_train, y_train).predict(x_test), y_test),
          recall_score(clf_lg.fit(x_train, y_train).predict(x_test), y_test),
          accuracy_score(clf_lg.fit(x_train, y_train).predict(x_test), y_test))

    svm_save_path = "svm_model.sav"

    rf_save_path = "random_forest.sav"

    lg_save_path = "logistic.sav"

    joblib.dump(svm_cv, svm_save_path)
    joblib.dump(gscv_rfc,rf_save_path)
    joblib.dump(clf_lg.fit(x_train, y_train),lg_save_path)

    print("models saved")













