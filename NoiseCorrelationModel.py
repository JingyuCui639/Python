# Module for different class of models for noise correlation
import numpy as np

#First, define a base class
class NoiseCorrelationModel():
	
	def __init__(self):
		
		self.parameter=None
		self.train_fitted_value=None
		self.train_r_sqr=None

	#For training set
	def fit(self, train_X, train_Y):

		raise NotImplementedError("Subclass must implement this abstract method")

	#For testing set
	def predict(self,test_X):

		raise NotImplementedError("Subclass must implement this abstract method")


	#For testing set
	def get_r_sqr(self, test_X,test_Y):

		fitted_Y=self.predict(test_X)
		RSS=sum((test_Y-fitted_Y)**2)
		TSS=sum((test_Y-np.mean(test_Y))**2)
		r_sqr=1-RSS/TSS
		return r_sqr

		#For testing set
	def get_r_sqr_againstIndependence(self, test_X,test_Y):

		fitted_Y=self.predict(test_X)
		RSS=sum((test_Y-fitted_Y)**2)
		TSS=sum((test_Y-0)**2)
		r_sqr=1-RSS/TSS
		return r_sqr

	def __str__(self):

		return f"parameter: \n{self.parameter}\n"\
		f"fitted response for training set: \n {self.train_fitted_value} \n"\
		f"r_sqr of the fit for training set: {self.train_r_sqr}"

	def summary(self):

		print(self)


#create a subclass "Noise Ceiling"
class NoiseCeiling(NoiseCorrelationModel):


	def fit(self,train_X,train_Y):

		self.train_fitted_value=train_Y

	def predict(self,test_X):

		return self.train_fitted_value

#Create a subclass "Linear Model"
class LinearModel(NoiseCorrelationModel):

	def fit(self, train_X,train_Y):

		X_transpose=np.transpose(train_X)
		self.parameter=np.linalg.inv(X_transpose@train_X)@X_transpose@train_Y
		self.train_fitted_value=train_X@self.parameter
		RSS=sum((train_Y-self.train_fitted_value)**2)
		TSS=sum((train_Y-np.mean(train_Y))**2)
		self.train_r_sqr=1-RSS/TSS

	def predict(self, test_X):

		return test_X@self.parameter


#Belows are some functions used for finding the betas by optimization
def linearModelPredict(b,X):
    yp = X@b
    return yp

def linearModelLossRSS(b,X,y):
    # Make predictions
    predY = linearModelPredict(b,X)
    # Compute residuals.  This is an array.  
    res = y-predY
    # Simply sum up the squared residuals.  This is the value of our loss.
    RSS= sum(res**2) 
    # Because res is a vector, we can take the product of res with X.
    gradient=-2*(res @ X)

    return (RSS, gradient)		

def linearModelFit(X,y,lossfcn = linearModelLossRSS):
    # Because we know b has to have the some dimension as X has columns,
    # We can use the number of columns to determine the size of betas
    # In this case, we use a 2d array
    nrows,ncols = X.shape
    betas=np.zeros((ncols,1))
    # Optimize the loss
    RES = so.minimize(lossfcn,betas,args=(X,y),jac=True)
    # Obtain estimates from the optimizer
    estimated_betas=RES.x 
    # Compute goodness of fit.
    res = y-np.mean(y)
    TSS = sum(res**2)
    RSS,deriv = linearModelLossRSS(estimated_betas,X,y)
    R2 = 1-RSS/TSS 

    return (estimated_betas,R2)


if __name__ == '__main__':

    print("NoiseCorrelationModel has been run directly!")
    	
else:

    print("NoiseCorrelationModel has been imported!")
