import os.path
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

models = [{
    'name': 'Logistic Regression',
    'model': LogisticRegression()
}, {
    'name': 'SVC',
    'model': SVC()
}, {
    'name': 'Linear SVC',
    'model': LinearSVC()
}, {
    'name': 'Naive Bayes',
    'model': GaussianNB()
}, {
    'name': 'KNN',
    'model': KNeighborsClassifier()
}, {
    'name': 'Decision Tree',
    'model': DecisionTreeClassifier()
}, {
    'name': 'Random Forest',
    'model': RandomForestClassifier()
}, {
    'name': 'Gradient Boosting',
    'model': GradientBoostingClassifier()
}, {
    'name': 'MLP',
    'model': MLPClassifier()
}]


def test_models(models, feature_names, features, label, cv, log):
    '''
    Runs a series of simple models to test Feature Engineering performance
    '''

    # Importing the log or creating one if it doesn't exist
    if os.path.isfile(log):
        df_log = pd.read_csv(log)
    else:
        df_log = pd.DataFrame(
            columns=['model', 'features', 'performance', 'cv']
        )

    # Running the models
    for item in models:

        name = item['name']
        scores = cross_val_score(item['model'], features, label, cv=cv)
        performance = round(scores.mean()*100, 2)
        performance_std = round(scores.std()*100, 2)
        performance_min = round(scores.min()*100, 2)
        performance_max = round(scores.max()*100, 2)

        # Output Performance
        print('{} Mean: {}%'.format(name, performance))
        print('{} STD: {}5'.format(name, performance_std))
        print('{} Min: {}%'.format(name, performance_min))
        print('{} Max: {}%'.format(name, performance_max))  

        # Log performance
        performance_dict = {
            'model': [item['name']],
            'features': [feature_names],
            'performance': [performance],
            'cv': [cv]
        }

        df_perf = pd.DataFrame.from_dict(performance_dict)
        df_log = pd.concat([df_log, df_perf])

    df_log = df_log.sort_values(by='performance', ascending=False)
    df_log.to_csv(log, index=False)
