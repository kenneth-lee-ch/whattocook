
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def spotcheck(X, y, foldnum , scoring, random_state):
	"""
		this function compare different classification algorithms without tuning the parameter by k-fold cross-validation

	Arguements:
		X (array): the training samples
		y (array):  the response variable
		foldnum (int): the number of fold
		scoring (str): the scoring method
		random_state (int): the seed number
	Return:
		None
	"""
	# prepare models
	models = []
	models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('RF', RandomForestClassifier()))

	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
	    kfold = StratifiedKFold(n_splits=foldnum, random_state=random_state)
	    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	    results.append(cv_results)
	    names.append(name)
	    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	    print(msg)
	# boxplot algorithm comparison
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()

def tune(X, y, model, grid, num_fold, scoring, random_state):
	"""
		this function wraps the methods for grid search cross validation for tuning parameters

	Arguements:
		X (array): the training samples
		y (array):  the response variable
		grid(dict): a dictionary contains information about which parameter to be tuned and the respective values involved.
		num_fold(int): the number of fold for cross-validation
		scoring (str): the scoring metric
	Return:
		grid_result (dict) : dict of numpy (masked) ndarrays. See sklearn.model_selection.GridSearchCV for details
	"""
	# Grid search cross validation
	kfold = StratifiedKFold(n_splits=num_fold, random_state=random_state)
	grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv= kfold, scoring=scoring)
	grid_result = grid_search.fit(X,y)
	# print the best result and respective parameter based on residual sum of squares 
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	return grid_result
