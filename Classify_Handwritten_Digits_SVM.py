import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib.cm as cm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

# #Create a subplot for five instances for the digits 0, 1 and 2.
# counter = 1
# for i in range(1,4):
# 	for j in range(1, 6):
# 		plt.subplot(3, 5, counter)
# 		plt.imshow(digits[(i-1)*8000+j].reshape((28,28)), cmap=cm.Greys_r)
# 		plt.axis('off')
# 		counter += 1
# plt.show()

if __name__ == '__main__':
	#Load the data
	data = fetch_mldata('MNIST original', data_home='data/mnist')
	
	#Scale the features and center each feature around the origin
	Features, Labels = data.data, data.target
	Features = Features/255.0*2 - 1

	#Split the data into training and test sets
	Features_train, Features_test, Labels_train, Labels_test = train_test_split(Features, Labels)

	#Instantiate an SVC object. 
	#'kernel' specifies the kernel to be used.
	#'C' controls regularization
	#'gamma' is the kernel coefficient for the sigmoid, polynomial, and RBF kernels.
	# Tune the hyperparameters using grid search.
	pipeline = Pipeline([('clf', SVC(kernel='rbf', gamma=0.01, C=100))])
	print Features_train.shape
	parameters = {
		'clf__gamma' : (0.01, 0.03, 0.1, 0.3, 1),
		'clf__C' : (0.1, 0.3, 1, 3, 10, 30)
	}
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, verbose=1, scoring='accuracy')
	grid_search.fit(Features_train[:10000], Labels_train[:10000])
	print 'Best score: %0.3f' %grid_search.best_score_
	print 'Best parameters set:'
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print '\t%s: %r' %(param_name, best_parameters[param_name])
	predictions = grid_search.predict(Features_test)
	print classification_report(Labels_test, predictions)