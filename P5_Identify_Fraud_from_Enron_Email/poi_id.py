#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
    




### Task 2: Remove outliers
### detect NA
NA = {}
for key, value in data_dict.iteritems():
    NA[key] = 0
    for y in value.values():
        print y
        if y == "NaN":
            NA[key] += 1

for key, value in sorted(NA.iteritems(),reverse=True, key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)

    
all_features = data_dict['ALLEN PHILLIP K'].keys()
print('Each person has %d features' %  len(all_features))
### detect NA in each feature
missing_values = {}
for x in all_features:
    missing_values[x] = 0
for person in data_dict.keys():
    records = 0
    for x in all_features:
        if data_dict[person][x] == 'NaN':
            missing_values[x] += 1
        else:
            records += 1

### Print results of completeness analysis
print('Number of Missing Values for Each Feature:')
for feature in all_features:
    print("%s: %d" % (feature, missing_values[feature]))    
    
    
## there are two wierd people "LOCKHART EUGENE E" with all NaN value and "THE TRAVEL AGENCY IN THE PARK" seems not a person
del data_dict["LOCKHART EUGENE E"] 
del data_dict["THE TRAVEL AGENCY IN THE PARK"] 
        

num_points = len(data_dict)
num_features = len(data_dict[data_dict.keys()[0]])

num_poi = 0
for x in data_dict.values():
    if x["poi"] == 1:
        num_poi += 1
print "Data_points:", num_points ," Features:" , num_features,  " POIs:", num_poi
for x in data_dict.values():
    plt.scatter(x['salary'] , x['bonus'] )
# there is an outlier which is total! delete it

del data_dict["TOTAL"]

# plot again
for x in data_dict.values():
    plt.scatter(x['salary'] , x['bonus'] )

### Task 3: Create new feature(s)

## create two new feature, proportion of people send message to poi and from poi

def message_proportion( poi_messages, other_messages ):
	if poi_messages== "NaN" or other_messages== "NaN":
		proportion = 0
	else:
		proportion = poi_messages/float(other_messages)
	return proportion
 
for name in data_dict:
    from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
    to_messages = data_dict[name]["to_messages"]
    proportion_from_poi = message_proportion( from_poi_to_this_person, to_messages )
    data_dict[name]["proportion_from_poi"] = proportion_from_poi
    from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
    from_messages = data_dict[name]["from_messages"]
    proportion_to_poi = message_proportion( from_this_person_to_poi, from_messages )
    data_dict[name]["proportion_to_poi"] = proportion_to_poi
    


 ### Task 1: Select what features you'll use.

features = data_dict[data_dict.keys()[0]].keys()
features.remove("poi")
features.remove("salary")
features.remove("email_address")

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] +  features# You will need to use more features
 
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,random_state = 47)

print "Intuitive features:", features_list


### run algorithn with all feature and see the result

#tree
clf = DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)
predict = clf.predict(features_test)
precision=  precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)
print precision
print recall

#SVM 
clf= svm.SVC()
clf = clf.fit(features_train,labels_train)
predict = clf.predict(features_test)
precision=  precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print precision
print recall

#Gaussian NB
clf = GaussianNB()
clf = clf.fit(features_train,labels_train)
predict = clf.predict(features_test)
precision=  precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print precision
print recall
# It looks like we can get better precision and recall by Gaussian NB
# but all of them are not good
# so we look if we do feature selection to get better result

## score function for precision, recall ,f1 by NB
def score_func(y_true,y_predict):
	true_negatives = 0
	false_negatives = 0
	true_positives = 0
	false_positives = 0
	for prediction, truth in zip(y_predict, y_true):
		if prediction == 0 and truth == 0:
			true_negatives += 1
		elif prediction == 0 and truth == 1:
			false_negatives += 1
		elif prediction == 1 and truth == 0:
			false_positives += 1
		else:
			true_positives += 1
	if true_positives == 0:
		return (0,0,0)
	else:
		precision = 1.0*true_positives/(true_positives+false_positives)
		recall = 1.0*true_positives/(true_positives+false_negatives)
		f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
		return (precision,recall,f1)
 
# K-best features for precision, recall ,f1 by NB
def selectKBest(previous_result, data):
	# remove 'restricted_stock_deferred' and 'director_fees' due to recall = 1 which overfit
	previous_result.pop(4)
	previous_result.pop(4)

	result = []
	_k = 15
	for k in range(0,_k):
		feature_list = ['poi']
		for n in range(0,k+1):
			feature_list.append(previous_result[n][0])

		data = featureFormat(my_dataset, feature_list, sort_keys = True, remove_all_zeroes = False)
		labels, features = targetFeatureSplit(data)
		features = [abs(x) for x in features]
		from sklearn.cross_validation import StratifiedShuffleSplit
		cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
		features_train = []
		features_test  = []
		labels_train   = []
		labels_test    = []
		for train_idx, test_idx in cv:
			for ii in train_idx:
				features_train.append( features[ii] )
				labels_train.append( labels[ii] )
			for jj in test_idx:
				features_test.append( features[jj] )
				labels_test.append( labels[jj] )
		from sklearn.naive_bayes import GaussianNB
		clf = GaussianNB()
		clf.fit(features_train, labels_train)
		predictions = clf.predict(features_test)
		score = score_func(labels_test,predictions)
		result.append((k+1,score[0],score[1],score[2]))
	return result
    
# each features importance for precision, recall ,f1 by NB    
def univariateFeatureSelection(f_list, my_dataset):
	result = []
	for feature in f_list:
		# Replace 'NaN' with 0
		for name in my_dataset:
			data_point = my_dataset[name]
			if not data_point[feature]:
				data_point[feature] = 0
			elif data_point[feature] == 'NaN':
				data_point[feature] =0

		data = featureFormat(my_dataset, ['poi',feature], sort_keys = True, remove_all_zeroes = False)
		labels, features = targetFeatureSplit(data)
		features = [abs(x) for x in features]
		from sklearn.cross_validation import StratifiedShuffleSplit
		cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
		features_train = []
		features_test  = []
		labels_train   = []
		labels_test    = []
		for train_idx, test_idx in cv:
			for ii in train_idx:
				features_train.append( features[ii] )
				labels_train.append( labels[ii] )
			for jj in test_idx:
				features_test.append( features[jj] )
				labels_test.append( labels[jj] )
		from sklearn.naive_bayes import GaussianNB
		clf = GaussianNB()
		clf.fit(features_train, labels_train)
		predictions = clf.predict(features_test)
		score = score_func(labels_test,predictions)
		result.append((feature,score[0],score[1],score[2]))
	result = sorted(result, reverse=True, key=lambda x: x[3])
	return result    
    
    
    
    
    
# univariate feature selection
features_list_nopoi = features_list
features_list_nopoi.remove("poi")
univariate_result = univariateFeatureSelection(features_list_nopoi,my_dataset)
print '### univariate feature selection result'
for x in univariate_result:
	print x

# select k best
select_best_result = selectKBest(univariate_result, my_dataset)
print '### select k best result'
for x in select_best_result:
	print x   
    
 
feature_best = ["poi"]
for i in univariate_result[:3]:
    print i[0]
    feature_best.append(i[0])
    

print feature_best 

data = featureFormat(my_dataset, feature_best , sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,random_state = 40)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### decision tree
tree = DecisionTreeClassifier(random_state = 42)

parameters = {'tree__criterion': ('gini','entropy'),
              'tree__splitter':('best','random'),
              'tree__min_samples_split':[2, 10, 20],
              'tree__max_depth':[10,15,20,25,30],
              'tree__max_leaf_nodes':[5,10,30]}
# use scaling in GridSearchCV
Min_Max_scaler = MinMaxScaler()
features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('tree', tree)])
cv = StratifiedShuffleSplit(labels, 100, random_state = 40)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, labels)
clf = gs.best_estimator_
print clf


clf.fit(features_train,labels_train)
predict = clf.predict(features_test)

precision=  precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)
print precision
print recall


### SVM

svc = svm.SVC()
parameters = {
 }
Min_Max_scaler = MinMaxScaler()
features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('SVM', svc)])
cv = StratifiedShuffleSplit(labels, 100, random_state = 40)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, labels)
clf = gs.best_estimator_
print clf

clf.fit(features_train,labels_train)
predict = clf.predict(features_test)

precision=  precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)
print precision
print recall


### GaussianNB

nb = GaussianNB()
parameters = {}
Min_Max_scaler = MinMaxScaler()
features = Min_Max_scaler.fit_transform(features)
pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('GaussianNB', nb)])
cv = StratifiedShuffleSplit(labels, 100, random_state = 40)

gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

gs.fit(features, labels)
clf = gs.best_estimator_
print clf

clf.fit(features_train,labels_train)
predict = clf.predict(features_test)

precision=  precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print precision
print recall





### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# we get the recall 0.398 and precision 0.493
  

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=40)
    
clf = GaussianNB()
clf = clf.fit(features_train,labels_train)
predict = clf.predict(features_test)

precision=  precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)
print precision
print recall

# looks like we still get not bad precision and recall
#test for my algorithm
from tester import test_classifier
print "Tester GaussianNB report" 
test_classifier(clf, my_dataset, feature_best) 

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
features_list = feature_best
dump_classifier_and_data(clf, my_dataset, features_list)

