import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.linear_model import LogisticRegression



# Ensure the OULAD dataset is in the same directory as this file before running
# Uncomment lines 84, 112, 113, 139, 144, 345, 385, 386 if you wish to view the data distributions in matplotlib
# Run with the command python classifier.py in the command line


#Store path to data directory as variable
data_dir = os.path.join(os.getcwd(), 'anonymisedData')

# read csv files
df_studentInfo = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
df_assessments = pd.read_csv(os.path.join(data_dir, 'assessments.csv'))
df_studentAssessment = pd.read_csv(os.path.join(data_dir, 'studentAssessment.csv'))
df_studentRegistration = pd.read_csv(os.path.join(data_dir, 'studentRegistration.csv'))
df_studentVle = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))
df_courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))
df_vle = pd.read_csv(os.path.join(data_dir, 'vle.csv'))



# merge presentation length with student info
df_studentInfo = df_studentInfo.merge(df_courses, 'inner', ['code_module', 'code_presentation'])

#Prepare Registration Data
# replace date_unregistered Nan values with 'NaN' if student has not unregisterd
df_studentRegistration = df_studentRegistration.fillna('NaN')
# set is_registered to 1 if student has not unregistered yet
df_studentRegistration['is_registered'] = np.where(df_studentRegistration['date_unregistration']=='NaN',1,0)
# select primary key and is_registered column and merge with student info dataframe
df_studentTotalregistration = df_studentRegistration[['id_student', 'code_module', 'code_presentation', 'is_registered']].groupby(by=['id_student', 'code_module', 'code_presentation']).sum()
df_studentInfo = df_studentInfo.merge(df_studentTotalregistration, 'left', ['id_student', 'code_module', 'code_presentation'])


#Prepare Assessments Data
# calculate mean score assignment score
df_studentAssessment['mean_score'] = df_studentAssessment['score'].groupby(by=df_studentAssessment['id_assessment']).transform('mean')
# subract mean score from students score
df_studentAssessment['mean_score_difference'] = df_studentAssessment['score'] - df_studentAssessment['mean_score']
# calculate mean score difference achieved by each student
df_studentAssessment = df_studentAssessment[['id_student', 'mean_score_difference']].groupby(by='id_student').mean()
# merge calculated difference with studentInfo dataframe
df_studentInfo = df_studentInfo.merge(df_studentAssessment, 'left', ['id_student'])

# Create column for each type of assessment and store as a binary variable
df_assessments['is_TMA_assessed'] = (np.where(df_assessments['assessment_type'] == 'TMA',1,0))
df_assessments['is_CMA_assessed'] = (np.where(df_assessments['assessment_type'] == 'CMA',1,0))
df_assessments['is_Exam_assessed'] = (np.where(df_assessments['assessment_type'] == 'Exam',1,0))
df_assessments = df_assessments[['code_module', 'code_presentation', 'is_TMA_assessed', 'is_CMA_assessed', 'is_Exam_assessed']].groupby(by=['code_module', 'code_presentation']).max()
# merge new columns with studentInfo dataframe
df_studentInfo = df_studentInfo.merge(df_assessments, 'left', ['code_module', 'code_presentation'])

#Prepare VLE dataset
# Merge studentVle dataframe with vle dataframe
df_studentVle = df_studentVle.merge(df_vle, 'left', ['id_site', 'code_module', 'code_presentation'])
# find mean number of sum clicks per activity type per student
df_vle = df_studentVle[['id_student', 'code_module', 'code_presentation', 'sum_click', 'activity_type']].groupby(['id_student', 'code_module', 'code_presentation', 'activity_type']).mean()
df_vle = df_vle.reset_index()

# function to plot number of clicks per activity
def plot_activity_clicks():
    vle_dist = df_vle['activity_type'].value_counts().plot(kind='bar',
							title='Activity Type Clicks',
                                                       rot=30)
    vle_dist.set_xlabel('Activity Type')
    vle_dist.set_ylabel('Count')
    plt.show()
# Uncomment following line to plot activity click distribution
#plot_activity_clicks()

# reshape vle dataframe
df_vle = df_vle.pivot_table(values='sum_click', columns='activity_type', index=['id_student', 'code_module', 'code_presentation'], fill_value=0)
df_vle = df_vle.reset_index()


# select foreign key student and activities from dataframe
df_vleActivities = df_vle[['code_module', 'code_presentation', 'id_student', 'externalquiz', 'forumng',
                 'glossary', 'homepage', 'oucollaborate', 'oucontent', 'ouwiki', 'page', 'quiz',
                 'resource', 'subpage', 'url']]

# if student didn't click on an activity, replace with 0 clicks
df_vleActivities = df_vleActivities.fillna(0)

# Merge vle activities with main dataframe
df_studentInfo = df_studentInfo.merge(df_vleActivities, 'left', ['id_student', 'code_module', 'code_presentation'])


# dropna for imd_band - has na values for categoric data
df_studentInfo = df_studentInfo.dropna(subset=['imd_band'])

# drop heavily skewed and irrelevant data from the dataset
df_studentInfo = df_studentInfo.drop(columns=['id_student', 'disability', 'num_of_prev_attempts', 'studied_credits', 'imd_band', 'subpage', 'ouwiki', 'is_TMA_assessed', 'is_CMA_assessed'])


# Plot histagrams to show distributions
#Uncomment following lines to see the histograms of data
#df_studentInfo.hist(bins=50, figsize=(20,15))
#plt.show()

# function to plot distribution for categorical variables
def plot_categorical_distributions():
    # plot final result distribution
    final_result_dist = df_studentInfo['final_result'].value_counts().plot(kind='bar',
							title='Final Result Distribution',
                                                       rot=0)
    final_result_dist.set_xlabel('Final Result')
    final_result_dist.set_ylabel('Count')
    # plot stacked distribution for how each other variable corresponds to final result
    pd.crosstab(df_studentInfo['num_of_prev_attempts'], df_studentInfo['final_result']).plot.bar(stacked=True, rot=0)
    plt.xlabel('Num of Prev Attempts')
    pd.crosstab(df_studentInfo['imd_band'], df_studentInfo['final_result']).plot.bar(stacked=True, rot=0)
    plt.xlabel('IMD BAND')
    pd.crosstab(df_studentInfo['gender'], df_studentInfo['final_result']).plot.bar(stacked=True, rot=0)
    plt.xlabel('Gender')
    pd.crosstab(df_studentInfo['disability'], df_studentInfo['final_result']).plot.bar(stacked=True, rot=0)
    plt.xlabel('Disabled')
    pd.crosstab(df_studentInfo['highest_education'], df_studentInfo['final_result']).plot.bar(stacked=True, rot=0)
    plt.xlabel('Highest Education')
    pd.crosstab(df_studentInfo['region'], df_studentInfo['final_result']).plot.bar(stacked=True, rot=0)
    plt.xlabel('Region')
    plt.show()

# Uncomment the following line to plot the distributions of the categorical variables
#plot_categorical_distributions()

# convert gender and disability to binary variable to avoid dummy variable trap
df_studentInfo['gender'] = (df_studentInfo['gender'] == 'M').astype(int)

# Train Test Split 70/30
train_set, test_set = train_test_split(df_studentInfo, test_size=0.2, random_state=42)

# Split labels from data
train_labels = train_set['final_result'].copy()
test_labels = test_set['final_result'].copy()

train_set = train_set.drop('final_result', axis=1).copy()
test_set = test_set.drop('final_result', axis=1).copy()



# DATA CLEANING

# split data columns into categoric and numeric
numeric_columns = train_set.select_dtypes(include=['int', 'float']).columns
categoric_columns = train_set.select_dtypes(exclude=['int', 'float']).columns
numeric_df = train_set[numeric_columns]
categoric_df = train_set[categoric_columns]



# replace na values with median and scale
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])


# convert categoric data to one hot encoding
full_pipeline = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_columns),
    ('categoric', OneHotEncoder(), categoric_columns),
])

train_data = full_pipeline.fit_transform(train_set)
test_data = full_pipeline.transform(test_set)

# TRAINING

# LOGISTIC REGRESSION

# initialise regressor for randomized search
logistic_regression_rs = LogisticRegression(multi_class='ovr', max_iter=2000)


# PARAMETER SEARCH
# set parameter options for randomized search
#regularization = ['l2']
# initialise grid for randomized search
random_grid = {
    #'penalty': regularization,
    'solver': ['lbfgs', 'newton-cg', 'saga'],
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'tol': [1e-6, 1e-5, 1e-4, 1e-3]
    }

# perform randomized search using f1 score with micro averaging
random_search_logistic_regression = RandomizedSearchCV(estimator=logistic_regression_rs,
                                                       param_distributions=random_grid,
                                                       n_iter=20,
                                                       scoring='neg_log_loss',
                                                       cv=3,
                                                       return_train_score=True,
                                                       verbose=1,
                                                       random_state=42,
                                                       n_jobs=-1
                                                       )

random_search_logistic_regression.fit(train_data, train_labels)
params = random_search_logistic_regression.best_params_

# initialise grid with C values +-10% of optimal C value from randomized search
param_grid = {
    'C': [params['C'] - params['C']/10, params['C'], params['C'] + params['C'] /10]
}
#penalty=params['penalty']
# initialise regressor for grid search
logistic_regression_gs = LogisticRegression(max_iter=2000, tol=params['tol'], multi_class='ovr', solver=params['solver'])

# initialise grid search for local optimal C value using 3 fold cross validation
logistic_regression_grid_search = GridSearchCV(logistic_regression_gs,
                                         param_grid,
                                         scoring='neg_log_loss',
                                         cv=3,
                                         return_train_score=True,
                                         n_jobs=-1)

# perform grid search
logistic_regression_grid_search.fit(train_data, train_labels)


logistic_regression_params = logistic_regression_grid_search.best_params_
logistic_regression_model = logistic_regression_grid_search.best_estimator_


# EVALUATION

# perform cross validation
lr_predictions = cross_val_predict(logistic_regression_model, train_data, train_labels, cv=3)

# calculate train data scores
lr_train_f1_score = f1_score(train_labels, lr_predictions, average='micro')
lr_train_cross_val_score = cross_val_score(logistic_regression_model, train_data, train_labels, cv=3)

#print train data scores
print('Logistic Regression Training f1 score:', lr_train_f1_score)
print('Logistic Regression Training mean cross validation score:', np.mean(lr_train_cross_val_score))

# results on test data
lr_test_predictions = logistic_regression_model.predict(test_data)
lr_test_f1_score = f1_score(test_labels, lr_test_predictions, average='micro')
lr_log_loss = log_loss(test_labels, logistic_regression_model.predict_proba(test_data))
lr_cross_val_score = cross_val_score(logistic_regression_model, test_data, test_labels, cv=3)

# print test data scores
print('Logistic Regression Test f1 score:', lr_test_f1_score)
print('Logistic Regression Test log loss:', lr_log_loss)
print('Logistic Regression Test mean cross validation score:', np.mean(lr_cross_val_score))

# generate confusion matrix
lr_test_confusion_matrix = confusion_matrix(test_labels, lr_test_predictions, labels=['Withdrawn', 'Pass', 'Distinction', 'Fail'])


# RANDOM FOREST MODEL

# PARAMETER SEARCH
# initialise 10 estimators for random search
n_estimators = [random.randint(11, 1000) for _ in range(10)]
# initialise bootstrap options for random search
bootstrap = [True]
# initialise max_features to number of features (auto) and square root of features (sqrt) for random search
max_features = ['auto', 'log2']
# initialise max depth
max_depth = [2, 4, 6, 8, 16, 18]
# initialise random grid
random_grid = {
    'n_estimators': n_estimators,
    'bootstrap': bootstrap,
    'max_features': max_features,
    'max_depth': max_depth
    }


# initialise random forest model for randomized search
random_forest_rs = RandomForestClassifier(oob_score=True)

# perform randomized search using random_grid using all processors on machine
# use F1 score to measure performance
# try 10 combinations and use 5 fold cross validation
random_search_random_forest = RandomizedSearchCV(estimator=random_forest_rs,
                                                 param_distributions=random_grid,
                                                 n_iter=20,
                                                 cv=3,
                                                 return_train_score=True,
                                                 verbose=1,
                                                 random_state=42,
                                                 n_jobs=-1)

random_search_random_forest.fit(train_data, train_labels)

# sort features by order of importance
attributes = df_studentInfo.columns
importances = random_search_random_forest.best_estimator_.feature_importances_
feature_importances = sorted(zip(attributes, importances), reverse=True)

params = random_search_random_forest.best_params_

# Create parameter grid with +-10% n_estimators of best estimator (nearest integer)
param_grid = {
    'n_estimators': [int(params['n_estimators'] - params['n_estimators']/10), params['n_estimators'], int(params['n_estimators'] + params['n_estimators'] / 10)]
}

# Perform grid search
random_forest_gs = RandomForestClassifier(bootstrap=params['bootstrap'], max_features=params['max_features'], oob_score=True, max_depth=params['max_depth'])
random_forest_grid_search = GridSearchCV(random_forest_gs,
                                         param_grid,
                                         cv=3,
                                         return_train_score=True,
                                         n_jobs=-1)

random_forest_grid_search.fit(train_data, train_labels)

random_forest_params = random_forest_grid_search.best_params_
random_forest_model = random_forest_grid_search.best_estimator_

# Function to plot feature importance
def plot_feature_importances():
    importances = random_forest_model.feature_importances_
    features = df_studentInfo.columns
    feature_importance = sorted(zip(features, importances), reverse=True)
    features = []
    importance = []
    for i, j in feature_importance:
        features.append(i)
        importance.append(j)
    dist = pd.Series(importance, index=features)
    dist.plot(kind='bar', title='Feature Importance', rot=90)
    plt.show()

# Uncomment the following line to plot feature importance
#plot_feature_importances()

# EVALUATION
# perform cross validation predictions
rf_predictions = cross_val_predict(random_forest_model, train_data, train_labels, cv=5)

# calculate train data scores
rf_train_f1_score = f1_score(train_labels, rf_predictions, average='micro')
rf_train_cross_val_score = cross_val_score(random_forest_model, train_data, train_labels, cv=3)

#print train data scores
print('Random Forest Training f1 score:', rf_train_f1_score)
print('Random Forest Training mean cross validation score:', np.mean(rf_train_cross_val_score))

# calculate test data scores
rf_test_predictions = random_forest_model.predict(test_data)
rf_test_f1_score = f1_score(test_labels, rf_test_predictions, average='micro')
rf_cross_val_score = cross_val_score(random_forest_model, test_data, test_labels, cv=3)

# print test data scores
print('Random Forest Test f1 score:', rf_test_f1_score)
print('Random Forest Test mean cross validation score:', np.mean(rf_cross_val_score))
print('Random Forest Test OOB score:', random_forest_model.oob_score_)

# generate confusion matrix
rf_test_confusion_matrix = confusion_matrix(test_labels, rf_test_predictions, labels=['Withdrawn', 'Pass', 'Distinction', 'Fail'])


# plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name):
    ticks = np.arange(4)
    plt.matshow(conf_matrix, cmap=plt.cm.gray)
    plt.xticks(ticks, ('Withdrawn', 'Pass', 'Distinction', 'Fail'))
    plt.yticks(ticks, ('Withdrawn', 'Pass', 'Distinction', 'Fail'), rotation='vertical')
    plt.ylabel('Final Result Prediction')
    plt.title(model_name)
    plt.show()

    
#Uncomment following lines to view the confusion matrices
#plot_confusion_matrix(lr_test_confusion_matrix, 'Logistic Regression')
#plot_confusion_matrix(rf_test_confusion_matrix, 'Random Forest')





