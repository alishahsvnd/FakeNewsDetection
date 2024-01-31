import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

def score(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    score = {
        'Precision':precision,
        'Recall':recall,
        'F1_Score':f1
         }
    # Return Precision, Recall, F1_Score
    return score

def run_cross_validation_on_trees(X, y, criterion, tree_depths, cv=10, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth, criterion=criterion)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores
  
# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1)
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()
    plt.show()

real = pd.read_csv('pol-real.csv')
fake = pd.read_csv('pol-fake.csv')

print('Real news:\n\n', real, '\n\n')
print('Fake news:\n\n', fake, '\n\n')

real['label'] = np.ones(len(real))
fake['label'] = np.zeros(len(fake))

total_data = pd.concat([real, fake])
print("Total data:\n\n", total_data, "\n\n")

X = total_data.loc[:, total_data.columns != 'label']
y = total_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
gini_clf = DecisionTreeClassifier(criterion='gini').fit(X_train, y_train)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(gini_clf,
                   filled=True)
fig.show()

print("Gini method accuracy: ",gini_clf.score(X_test, y_test), '\n\n')

gini_pred = gini_clf.predict(X_test)
gini_cm = confusion_matrix(y_test, gini_pred)
print('Gini methdo confusion matrix: ', gini_cm,'\n\n')
print("Gini method Precision, Recall and F1_score:\n\n", score(gini_cm), '\n\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
info_clf = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(info_clf,
                   filled=True)
fig.show()

print("Information gain method accuracy: ",info_clf.score(X_test, y_test), '\n\n')

info_pred = info_clf.predict(X_test)
info_cm = confusion_matrix(y_test, info_pred)
print('Information gain method confusion matrix: ', info_cm,'\n\n')
print("Information gain method Precision, Recall and F1_score:\n\n", score(gini_cm), '\n\n')

sm_tree_depths = range(5,20)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X, y, "gini", sm_tree_depths)

# plotting accuracy
plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
                               'Accuracy per decision tree depth on training data')
idx_max = sm_cv_scores_mean.argmax()
sm_best_tree_depth = sm_tree_depths[idx_max]
sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
      sm_best_tree_depth, round(sm_best_tree_cv_score*100,5), round(sm_best_tree_cv_score_std*100, 5)))

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
gini_optimal = DecisionTreeClassifier(criterion='gini', max_depth=sm_best_tree_depth).fit(X_train, y_train)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(gini_optimal,
                   filled=True)
fig.show()

print('Optimal gini accuracy: ', gini_optimal.score(X_test, y_test), '\n\n')

gini_optimal_pred = gini_optimal.predict(X_test)
gini_optimal_cm = confusion_matrix(y_test, gini_optimal_pred)
print("Optimal gini confusion matrix:\n\n",gini_optimal_cm,'\n\n')
print("Optimal gini Precision, Recall, F1_Score:\n\n", score(gini_optimal_cm),'\n\n')

sm_tree_depths = range(5,20)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X, y, "entropy", sm_tree_depths)

# plotting accuracy
plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
                               'Accuracy per decision tree depth on training data')
idx_max = sm_cv_scores_mean.argmax()
sm_best_tree_depth = sm_tree_depths[idx_max]
sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
      sm_best_tree_depth, round(sm_best_tree_cv_score*100,5), round(sm_best_tree_cv_score_std*100, 5)))

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
info_optimal = DecisionTreeClassifier(criterion='entropy', max_depth=sm_best_tree_depth).fit(X_train, y_train)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(info_optimal,
                   filled=True)
fig.show()

print('Optimal info method accuracy: ', info_optimal.score(X_test, y_test
), '\n\n')

info_optimal_pred = info_optimal.predict(X_test)
info_optimal_cm = confusion_matrix(y_test, info_optimal_pred)
print('Optimal info confusion matrix:\n\n', info_optimal_cm, '\n\n')
print('Optimal info Precision, Recall, F1_Score:\n\n', score(info_optimal_cm), '\n\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
random_clf = RandomForestClassifier().fit(X_train, y_train)

print('Random Forest accuracy: ', random_clf.score(X_test, y_test), '\n\n')

random_pred = random_clf.predict(X_test)
random_cm = confusion_matrix(y_test, random_pred)
print('Random forest confusion matrix:\n\n', random_cm, '\n\n')

print('Random forest Precision, Recall, F1_score:\n\n', score(random_cm), '\n\n')