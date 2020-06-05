
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
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout,  BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
from keras import backend as K


def spotcheck(X, y, models, foldnum , scoring, random_state):
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

	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		# Perform stratified k-fold
	    kfold = StratifiedKFold(n_splits=foldnum, random_state=random_state)
	    # Cross-validate based on specified stratified k-fold
	    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	    # add result to a list 
	    results.append(cv_results)
	    names.append(name)
	    # Compute the mean and standard deviation of thhe score
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
	# Utilize all CPUs and assign 0 score if the error occurs. 
	grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=4, cv= kfold, scoring=scoring, error_score=0)
	grid_result = grid_search.fit(X,y)
	# print the best result and respective parameter based on residual sum of squares 
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	return grid_result


def build_model(node_num, bn, activation1, activation2, hidden_layers, optimizer, init, n_features):
    """
        create a model based on specified parameters and use this function along with Kerasclassifier() to build architectures pipelines

    Arguments:
    	node_num (int): the number of nodes in the hidden layer
    	bn (bool): determine whether we want to add batch normalization layer
    	activation1 (str): activation function for the first hidden layer
    	activation2 (str): activation for the hidden layers after the first hidden layer
    	hidden_layers (int): the number of hidden layers to be added besides the first hidden layer
    	optimizer (str): the name of the optimizer to train the model
    	init (str): the weight initalization
    	n_features (int): the input dimension
    Return:
    	model : a neural network model

    """
    K.clear_session()
    # create model
    model = Sequential()
    # Check if we need embedding layer
    model.add(Dense(node_num, input_dim = n_features, kernel_initializer=init, activation=activation1))
    for i in range(int(hidden_layers)):
        # determine if we need to add batch normalization layer
        if bn == True:
        	# Add batch normalization layer
            model.add(BatchNormalization())
            # Add hidden layer
        model.add(Dense(node_num, kernel_initializer=init, activation=activation2))
    # Output 20 different targets
    model.add(Dense(20, kernel_initializer=init, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    return model
