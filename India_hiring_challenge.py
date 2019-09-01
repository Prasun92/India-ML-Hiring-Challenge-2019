# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:44:29 2019

@author: prasu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

data = pd.read_csv(r'train.csv')
test = pd.read_csv(r'test.csv')

#print(data.head())
print('Missing value check')
print(data.isnull().sum())
print(test.isnull().sum())

print(data.m13.value_counts())

print('Elporatory Data Analysis')

plt.hist(data['interest_rate'])
plt.show()
plt.hist(data['unpaid_principal_bal'])
plt.show()
plt.hist(data['unpaid_principal_bal'], log= True)
plt.show()
plt.hist(data['loan_term'])
plt.show()
plt.hist(data['loan_to_value'])
plt.show()
plt.hist(data['loan_to_value'], log = True)
plt.show()
plt.hist(data['debt_to_income_ratio'])
plt.show()
plt.hist(data['borrower_credit_score'])
plt.show()
plt.hist(data['borrower_credit_score'], log = True)
plt.show()

print('Data Wrangling')
data['loan_to_value'] = data['loan_to_value'].apply(np.log)
data['unpaid_principal_bal'] = data['unpaid_principal_bal'].apply(np.log)

test['loan_to_value'] = test['loan_to_value'].apply(np.log)
test['unpaid_principal_bal'] = test['unpaid_principal_bal'].apply(np.log)

data_corr = data[['financial_institution','interest_rate','unpaid_principal_bal','loan_term','loan_to_value','number_of_borrowers','debt_to_income_ratio','borrower_credit_score','insurance_percent','co-borrower_credit_score','insurance_type']]

corr = data_corr.corr()
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())
    
sns.heatmap(data_corr.corr(), cmap='BuGn')
plt.show()

month_corr = data[['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13']]
sns.heatmap(month_corr.corr(), cmap='BuGn')
plt.show()

data_initial = data[['financial_institution', 'interest_rate',
       'unpaid_principal_bal', 'loan_term', 'loan_to_value', 'number_of_borrowers',
       'debt_to_income_ratio', 'borrower_credit_score', 'loan_purpose',
       'insurance_percent', 'co-borrower_credit_score', 'insurance_type', 'm1',
       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
       'm13']]

data_test = test[['financial_institution', 'interest_rate',
       'unpaid_principal_bal', 'loan_term', 'loan_to_value', 'number_of_borrowers',
       'debt_to_income_ratio', 'borrower_credit_score', 'loan_purpose',
       'insurance_percent', 'co-borrower_credit_score', 'insurance_type', 'm1',
       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']]

from sklearn.preprocessing import LabelEncoder

column_name = ['financial_institution','loan_purpose']
lb = {}
for x in column_name:
    lb[x] = LabelEncoder()
    
for x in column_name:
    lb[x].fit(data_initial[x])
    data_initial[x] = lb[x].transform(data_initial[x])
    data_test[x] = lb[x].transform(data_test[x])

#print(data_initial.head())

X_train = data_initial.values[:,:-1]
Y_train = data_initial.values[:,-1]

from sklearn.preprocessing import StandardScaler
#
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler1.fit(X_train)
scaler2.fit(data_test)
initial_train = scaler1.transform(X_train)
initial_test = scaler2.transform(data_test)

col_name = ['financial_institution', 'interest_rate',
       'unpaid_principal_bal', 'loan_term', 'origination_date',
       'first_payment_date', 'loan_to_value', 'number_of_borrowers',
       'debt_to_income_ratio', 'borrower_credit_score', 'loan_purpose',
       'insurance_percent', 'co-borrower_credit_score', 'insurance_type', 'm1',
       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
       'm13']
def sortSecond(val):
    return val[1]

#print("ADABOOST CLASSIFIER FEATURE IMPORTANCE")
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#abc_fea = AdaBoostClassifier(base_estimator=RandomForestClassifier(),n_estimators=10)
#abc_fit = abc_fea.fit(initial_train,Y_train)
#list1 = list(zip(col_name, abc_fit.feature_importances_))
#list1.sort(key=sortSecond, reverse= True)
#print("ADABOOST CLASSIFIER FEATURE IMPORTANCE", list1)
#
#
#print("FEATURE SELECTION USING DECISION TREE")
##from sklearn.tree import DecisionTreeClassifier
#feat_DT = DecisionTreeClassifier(criterion="gini", min_samples_split= 4)
#DT = feat_DT.fit(initial_train, Y_train)
#list3 = list(zip(col_name, feat_DT.feature_importances_))
#list3.sort(key=sortSecond, reverse= True)
#print("FEATURE SELECTION USING DECISION TREE: ", list3)

#important_feature = ['financial_institution', 'interest_rate',
#       'unpaid_principal_bal', 'loan_term','number_of_borrowers','loan_to_value',
#       'debt_to_income_ratio', 'borrower_credit_score','loan_purpose','m4', 'm6', 'm9','m12']
#
#train_imp = data_initial[['financial_institution', 'interest_rate',
#       'unpaid_principal_bal', 'loan_term','number_of_borrowers','loan_to_value',
#       'debt_to_income_ratio', 'borrower_credit_score','loan_purpose','m4', 'm6', 'm9','m12','m13']]
#test_imp = data_test[['financial_institution', 'interest_rate',
#       'unpaid_principal_bal', 'loan_term','number_of_borrowers','loan_to_value',
#       'debt_to_income_ratio', 'borrower_credit_score','loan_purpose','m4', 'm6', 'm9','m12']]
#
#data_X = train_imp.values[:,:-1]
#data_Y = train_imp.values[:,-1]
#
#
#scaler1 = StandardScaler()
#scaler2 = StandardScaler()
#scaler1.fit(data_X)
#scaler2.fit(test_imp)
#X_train = scaler1.transform(data_X)
#X_test = scaler2.transform(test_imp)

from sklearn.model_selection import train_test_split

#X_train, X_test_test, Y_train, Y_test_test = train_test_split(X_train, data_Y, test_size = 0.2, random_state = 42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

classifer = (LogisticRegression())
#classifer.fit(X_train,Y_train)
#Y_pred=classifer.predict(X_test)
#print("Logistic Regression")
#print(confusion_matrix(Y_test,Y_pred))
#print(accuracy_score(Y_test, Y_pred))
#print(classification_report(Y_test,Y_pred))

#print("RANDOM FOREST CLASSIFIER")
#from sklearn.ensemble import RandomForestClassifier
#model_RandomForest = RandomForestClassifier(n_estimators = 50, min_samples_split  = 5, class_weight = 'balanced_subsample', random_state  = 42)
#model_RandomForest.fit(X_train,Y_train)
#Y_pred=model_RandomForest.predict(X_test)


#print('ADABOOST CLASSIFIER')
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#model_AdaBoost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=50)
### fit the model on the data and predict the values
#model_AdaBoost.fit(X_train,Y_train)
#Y_pred=model_AdaBoost.predict(X_test)

#print('Randominized Search CV')
#from sklearn.model_selection import RandomizedSearchCV
#
#C = [0.001, 0.01, 0.1, 1.0, 10]
#class_weight = ['balanced',None]
#max_iter = [75,100,125,150,175,200]
#multi_class = ['ovr', 'multinomial', 'auto']
#random_state = 42
#solver = ['lbfgs','sag','saga']
#penalty = ['l2']
#
#random_grid = {'C':C,
#               'class_weight':class_weight,
#               'max_iter':max_iter,
#               'multi_class':multi_class,
#               'solver': solver,
#               'penalty':penalty}
#
#rfc = LogisticRegression(random_state=42)
#random_seach = RandomizedSearchCV(estimator=rfc,
#                                  param_distributions = random_grid,
#                                  n_iter = 50,
#                                  scoring = 'accuracy',
#                                  cv = 5,                                  
#                                  verbose = 1,
#                                  random_state = 42  )
#random_seach.fit(X_train, Y_train)
#
#print('The best hyperparameter for Random Search')
#print(random_seach.best_params_)
#print('The best score ')
#print(random_seach.best_score_)
#
#best_rfc = random_seach.best_estimator_
#print(best_rfc)
#best_rfc.fit(X_train,Y_train)
#Y_pred = best_rfc.predict(X_test)

#The best hyperparameter for Random Search
#{'solver': 'saga', 'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 75, 'class_weight': None, 'C': 0.1}
#The best score 
#0.996045080907822
#LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=75, multi_class='ovr', n_jobs=None,
#          penalty='l2', random_state=42, solver='saga', tol=0.0001,
#          verbose=0, warm_start=False)

#print("USING RECURSIVE FEATURE SELECTION")
#from sklearn.feature_selection import RFE
#log_class = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=75, multi_class='ovr', n_jobs=None,
#          penalty='l2', random_state=42, solver='saga', tol=0.0001,
#          verbose=0, warm_start=False)
#rfe = RFE(log_class)
#model_rfe = rfe.fit(initial_train, Y_train)
#print("Num Features: ", model_rfe.n_features_)
#print(" Selected Feature ")
#print(list(zip(col_name, model_rfe.support_)))
#
#ranking = model_rfe.ranking_
##print("Feature Ranking: ", model_rfe.ranking_)
#print("Feature Ranking: ", ranking)
#Y_pred = model_rfe.predict(initial_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import RepeatedKFold

#kfold = RepeatedKFold(random_state = 42)
#for train, test in kfold.split(X_train, Y_train):
#    model_seq = Sequential()
#    model_seq.add(Dense(units = 52, activation='relu',input_shape = (13,)))
#    #model_seq.add(Dropout(0.2))
#    model_seq.add(Dense(26,activation='relu'))
#    #model_seq.add(Dropout(0.2))
#    model_seq.add(Dense(1, activation='softmax'))
#    model_seq.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])
#    model_seq.summary()
#    model_seq.fit(X_train[train],Y_train[train],batch_size = 100,epochs = 10, verbose = 1)
##    myloss = model_seq.history.history['loss']
##    plt.plot(range(len(myloss)),myloss)

import xgboost as xgb

#dtrain = xgb.DMatrix(data = X_train, label= Y_train)
#dtest = xgb.DMatrix(data = X_test_test)
#dtrain = xgb.DMatrix(data = initial_train, label= Y_train)
#dtest = xgb.DMatrix(data = initial_test)

#params = {'objective' : 'binary:hinge',
#          'eval_metric' : 'error@0.4',
#          'verbosity':2,
#          'eta' : 0.1,
#          'learning_rate' : 0.03,
#          'gamma' : 0.75,
#          'max_depth' : 6,
#          'min_child_weight' : 7,
#          'max_delta_step' : 2,
#          'subsample' : 1,
#          'colsample_bytree' : 0.5,
#          'lambda' : 1,
#          'alpha' : 0,
#          'n_estimators' : 500}
#num_boost_round = 32
#
#bst = xgb.train(params, dtrain, num_boost_round = num_boost_round)

xgb_model = xgb.XGBClassifier(base_score=0.3, booster='dart', colsample_bylevel=0.7,
       colsample_bynode=1, colsample_bytree=0.8, gamma=7,
       learning_rate=0.01, max_delta_step=8, max_depth=4,
       min_child_weight=2, missing=None, n_estimators=400, n_jobs=1,
       nthread=None, objective='binary:hinge', random_state=42,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=2)
xgb_model.fit(initial_train,Y_train)

Y_pred = xgb_model.predict(initial_test)


#Y_pred = model_seq.predict(X_test)
prediction = pd.DataFrame(Y_pred, columns = ['Y_Pred'])
#prediction['Y_binary'] = np.where(prediction.Y_Pred > 0.63, 1,0)

#model_seq.save('E:\Office\Data Science\Hackerearth\India ML hiring Hackathon 2019\Model_seq.h5')

lg_sub = pd.DataFrame({'loan_id':test['loan_id'],'m13':Y_pred})
lg_sub = lg_sub[['loan_id','m13']]
lg_sub.to_csv("SubmissionXGBoost.csv",index = False)

