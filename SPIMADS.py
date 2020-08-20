'''
SPIMADS is a set of tool for design optimisation using lengthy and expensive black box evaluations.
Initially used for parameter optimisation using the computational fluid dynamics package STAR CCM+ 
as the black box (>2hour/evaluation), SPIMADS focusses on minimising costly function evaluations
and being able to flexibly assign them to resources. This method is far from optimal for solving the 
trival function provided as a test however! - Tim Attwell

n.b.
- Using anaconda
- Required: pip install pykriging
- Noticed a bug with latest version of scipy(1.5) in linux causing a false out of bounds error ~60% of 
  the time. Please downgrade to scipy 1.4.1 to fix.
'''
import numpy as np
#import scipy
#from scipy.optimize import minimize
import copy
from matplotlib import pyplot as plt
import pylab
from math import inf
import pyKriging
from pyKriging.regressionkrige import regression_kriging
from pyKriging.samplingplan import samplingplan
from pyKriging.krige import kriging
from mpl_toolkits.mplot3d import axes3d
from random import Random
from time import time
import math as m
#import bb          # Import a wrapper for a black box if required

try:
    import cPickle as leek
except ImportError:
    import pickle as leek


def optFunct(X):
    '''
    optFunct() is the function to be optimised. For this test case it 
    merely checks as a wrapper for a simple z = f(x,y) equation, but 
    this could also be used to wrap a much more complex call to a 
    black box software package. Can accept multiple evaluation points 
    within X.
    '''
    # Format list inputs to be np.array()
    try:
        X.shape[1]
    except:
        X = np.array([X])
    
    # Check that input is 2 dimensional (todo: make flexible dimensionality)
    if X.shape[1] != 2:
        raise Exception
    
    # Check over the searched locations in "eval_hist", and if a solution point
    # has already been calculated, skip actual function evaluation and replace with 
    # saved result.
    Xchecked=[]
    Xcopy=[]
    ycopy=[]
    copylist=[]
    for x in list(X):
        copycheck = False
        c=0
        for prev in optim.eval_hist['location']:
            # Check over previous results for evaluations of current point
            if all(x == prev):
                copycheck = True
                Xcopy.append(list(x))
                ycopy.append(list(optim.eval_hist['value'])[c])
                #print("Copy!")
                copylist.append(True)
                break
            c=c+1
        # If no previous points are found, mark location for evaluation
        if copycheck == False:
            Xchecked.append(list(x))
            copylist.append(False)

    # Input un-evaluated points through the function
    X = np.array(Xchecked)
    if X.any():
        # Modified Rosenbrock function with minimum y=0 at X=[0.5,0.5]
        x0 = 2*X[:,0]
        x1 = 2*X[:,1]
        a = 1
        b = 1000
        y = (a - x0)**2 + b*(x1-x0**2)**2
        # y = bb.bb(x0,x1) # Blackbox function can be called via 'bb'
    else:
        y=np.array([])

    # Put the newly evaluated and historically retrieved values
    # togeter in the order that the points were passed into optFunc(X).
    yfinal=[]
    xfinal=[]
    d=0
    f=0
    for ii in range(len(y)+len(ycopy)):
        if copylist[ii] == True:
            yfinal.append(ycopy[d])
            xfinal.append(Xcopy[d])
            d=d+1
        elif copylist[ii] == False:
            yfinal.append(y[f])
            xfinal.append(X[f,:])
            f=f+1
    
    # Return correnctly ordered solutions in the form of a np.array()
    y=np.array(yfinal)
    return y

class MADS_Optimiser():
    '''
    The Optimiser class contains the mechanisms for both the search and poll steps.
    It is initialised with the required dimensionality, minimum mesh size, function
    to be optimised, and several other parameters to define the search properties. 
    Currently not as flexible as I would like it to be
    '''
    def __init__(self, dim=2 ,e=1/64, testfunction=optFunct, testPoints=None, delt=1, tau=4, regswitch=False, opp=False, **kwargs):
        self.regressionswitch = regswitch       # Tell the model to use a regression based kriging model or not
        self.n = copy.deepcopy(dim)
        self.y = inf                            # Initialise y
        self.y_min_GLOBAL = inf
        self.delt = copy.deepcopy(delt)         # Frame size parameter (0,inf)
        self.ds = copy.deepcopy(delt)           # Mesh size parameter (0,inf)
        self.tau = copy.deepcopy(tau)           # Mesh size adjustment parameter (0,1; rational)
        self.G = np.eye(2)                      # Invertiuble matrix defining mesh 'shape'.
        self.Z = np.array([[1,0,-1],[0,1,-1]])  # Positive spanning set for R^n
        self.D = self.G.dot(self.Z)             # Set of directions in R^n
        self.e = copy.deepcopy(e)               # Minimum mesh size parameter

        self.testfunction = testfunction        # Define function to be optimised

        self.opp = opp                          # Switch to opportunistic search algorithm

        self.ii = 0                             # Initialise iteration counter
        self.term = False                       # Initialise TERMINATE flag to false

        # Initialise history and tracking variable able to prevent repeat evaluations,
        # restart simuations that have crashed. and review the progress of the optimiser.

        self.history = {}
        self.history['location'] = []
        self.history['value'] = []
        self.history['step_type'] = []
        self.history['iteration'] = []

        # History of all function evaluations
        self.eval_hist = {}
        self.eval_hist['location'] = []
        self.eval_hist['value'] = []

        # History of evaluations passed to kriging model
        self.krig_hist = {}
        self.krig_hist['location'] = []
        self.krig_hist['value'] = []

        # Record of the evaluations of the initial Latin 
        # Hypercube Sampling set.
        self.lhs_record = {}
        self.lhs_record['location'] = []
        self.lhs_record['value'] = []

    # Main Functions are:
    #     parameter_update()
    #     search()
    #     poll()

    # Utility Functions are:
    #     termination()
    #     rand_vector()
    #     iter()
    #     poll_directions()
    
    def parameter_update(self):
        '''
        Finds mesh size based on frame size
        '''
        self.ds = min([self.delt, self.delt**2])


    def search(self):
        '''
        Search all mesh points over the parameter space. The initial search
        uses quickly obtained prediction from the kriging suyrrogate model.
        The three lowest predicted locations are then evaluated using optFunc()
        '''
        # Try searching all grid points (for now)
        Mk = np.linspace(-1, 1, min([int(2/self.ds)+1,int(2*32)+1]))
        # todo - once certain number of mesh points reached, use LHS to search 
        # a subset of total mesh points.

        # Create list of all mesh points to evaluate
        xe=[]
        for ii in range(len(Mk)):
            for jj in range(len(Mk)):
                xe.append([Mk[ii],Mk[jj]])
        # Evaluate surrogate function at all points on Mk^n
        ye = []
        for ii in range(len(xe)):
            ye.append(self.krig.predict(xe[ii]))

        # Sort [evaluations, locations] from smallest eval to largest eval
        yxe=sorted(zip(ye,xe))

        # Find the '3' lowest estimates, while also ensuring no two selections 
        # are within a certain radius of each other. CLUNKY ALGORITHM - todo: improve
        threesmall=[yxe[0]]
        c=1
        while len(threesmall) < 3:
            pointdist=[]
            if len(threesmall) == 1:
                for ii in range(self.n):
                    pointdist.append(yxe[c][1][ii]-threesmall[0][1][ii])
                pointdist=np.array(pointdist)
                if np.linalg.norm(pointdist) > 0.5:
                    threesmall.append(yxe[c])

            pointdist1=[]
            pointdist2=[]
            if len(threesmall) == 2:
                for ii in range(self.n):
                    pointdist1.append(yxe[c][1][ii]-threesmall[0][1][ii])
                    pointdist2.append(yxe[c][1][ii]-threesmall[1][1][ii])
                pointdist1=np.array(pointdist1)
                pointdist2=np.array(pointdist2)
                if ((np.linalg.norm(pointdist1) > 0.5) & (np.linalg.norm(pointdist2) > 0.5)):
                    threesmall.append(yxe[c])
            c=c+1

        # Evaluate the selected points in optFunc()
        ysearchX=[]
        ysearche=[]
        for ii in range(len(threesmall)):
            ysearchX.append(threesmall[ii][1])
            ysearche.append(threesmall[ii][0])
        print("Search Point: "+str(ysearchX))
        ysearch = self.testfunction(np.array(ysearchX))
        
        # Find the smallest real evaluation
        ysearchmin = min(ysearch)
        # ...and find it's location
        for ii in range(len(ysearch)):
            if ysearch[ii] == ysearchmin:
                xsearchmin = ysearchX[ii][:]
                break

        # Check if a new global minimum has been found, and if so
        # update the global tracker
        if ysearchmin < self.y_min_GLOBAL:
            self.y_min_GLOBAL = ysearchmin
            self.X = xsearchmin
            self.delt = min([1,self.tau*self.delt])
            print("+++++Search success+++++")
            self.st = 's'
            self.searchSuccess=True
        else:
            self.searchSuccess=False
            print("///Search failiure - moving to Poll///")

        # Add evaluated points to kriging model and records
        for pp in range(len(ysearch)):
            iscopy=False
            for cc in range(len(self.eval_hist['location'])):
                if np.all(ysearchX[pp] == self.eval_hist['location'][cc]):
                    iscopy = True
                    break
            if (ysearch[pp] != inf) & (iscopy == False):
                print(('Adding points (s): {}\nWith value (s): {}'.format(ysearchX[pp],ysearch[pp])))
                self.krig.addPoint(ysearchX[pp], ysearch[pp])
                self.krig_hist['location'].append(copy.deepcopy(ysearchX[pp]))
                self.krig_hist['value'].append(copy.deepcopy(ysearch[pp]))
                self.eval_hist['location'].append(copy.deepcopy(ysearchX[pp]))
                self.eval_hist['value'].append(copy.deepcopy(ysearch[pp]))

        # Train kriging model with new points
        if self.searchSuccess==True:
            self.krig.train()
            self.krig.snapshot()

    
    def poll(self):
        '''
        Searches points around the current global minimum, looking for incremental improvements.
        Implements the Mesh Adaptive Direct Search algorithm which uses randomised search 
        vectors to mesh points within a set search frame, or boundary, around the current 
        global min.
        '''
        # Find set of n+1 poll points based on random poll_directions() function
        self.poll_directions()
        self.P = (self.D*self.ds) + np.repeat(self.X, self.D.shape[1]).reshape(self.Z.shape)

        # The opportunistic method will evaluate each point in sequence and move
        # to a new minima as soon as it is found. The non-opportunistic method
        # evaluates all poll points, and compares the poll minimum to the global minimum.
        if self.opp == True:
            pollsucc = False
            yPolMin = inf
            # Evaluate poll points one after the other
            for xPol in self.P.T:
                print("Poll Point: "+str(xPol))
                iscopy=False
                # If a previous evaluation at this point exists, use historical result...
                for cc in range(len(self.eval_hist['location'])):
                    if np.all(xPol == self.eval_hist['location'][cc]):
                        iscopy = True
                        yPol = self.eval_hist['value'][cc]
                        break
                # ... if not, evaluate and add to eval_hist (provided it is within boundaries[-1.1])
                if ((iscopy == False) & (any(abs(xPol) <= 1))):
                    yPol = self.testfunction(xPol)
                
                    self.eval_hist['location'].append(list(copy.deepcopy(xPol)))
                    self.eval_hist['value'].append(copy.deepcopy(yPol))
                # Barrier function - sets points outside of constraint to inf
                else:
                    yPol = inf

                # If new poll minumm found, update poll minimum
                if yPol < yPolMin:
                    yPolMin = yPol

                # Check if a new global minimum has been found, and if so,
                # break out of poll loop.
                if yPolMin < self.y_min_GLOBAL:
                    self.X = copy.deepcopy(xPol)
                    self.delt = min([self.delt * self.tau,1])
                    self.y_min_GLOBAL = yPolMin
                    #lineSearch = True      #future implementation of line search checking
                    pollsucc = True
                    print("+++++Poll success+++++")
                    self.st = 'p'
                    break
                    
            # If no new minima has been found, refine frame and mesh parameters
            if pollsucc == False:
                yPolMin = inf
                self.X = self.X
                self.delt = self.delt / self.tau
                print("-----S&P failiures, refining search.-----")
                self.st = 'f'
                
        else:
            # Evaluates all poll points using funtion
            yPol = self.testfunction(self.P.T)
            yPol=np.array(yPol)
            print(self.P.T)
            print(yPol)

            # Evaluations are saved
            for ii in range(len(yPol)):
                self.eval_hist['location'].append(copy.deepcopy(self.P.T[ii]))
                self.eval_hist['value'].append(copy.deepcopy(yPol[ii]))

            # Barrier function - sets points outside of constraint to inf
            for jj in range(len(yPol)):
                if any(abs(self.P[:,jj]) > 1):
                    yPol[jj] = inf

            # Finds smallest evaluation
            yPolMin = min(yPol)
            # ...and its location
            xPolMin = np.where(yPol == yPolMin)
            xPol = self.P[:,xPolMin].T[0][0]

            # Check if a new global minimum has been found
            if yPolMin < self.y_min_GLOBAL:
                self.X = xPol
                self.delt = min([self.delt * self.tau,1])
                self.y_min_GLOBAL = yPolMin
                #lineSearch = True
                print("+++++Poll success+++++")

            # If no new minima has been found, refine frame and mesh parameters
            else:
                self.X = self.X
                self.delt = self.delt / self.tau
                print("-----S&P failiures, refining search.-----")
            
        # Add evaluated points to the kriging model   
        iscopy=False
        for cc in range(len(optim.krig_hist['location'])):
            if all(xPol == optim.krig_hist['location'][cc]):
                iscopy = True
                break
        # Disallows inf evaluations to enter the kriging model
        if (yPolMin != inf) & (iscopy == False):
            print(('Adding points (p): {}\nWith value (p): {}'.format(xPol,yPolMin)))
            self.krig.addPoint(xPol, yPolMin)
            self.krig_hist['location'].append(copy.deepcopy(xPol))
            self.krig_hist['value'].append(copy.deepcopy(yPolMin))
        # Update kriging model
        self.krig.train()
        self.krig.snapshot()


    def startPoint(self, l, u):
        '''
        Defines a starting point within a range. Only used if not running a search step.
        '''
        n = len(l)
        x = np.zeros(n)
        x[0] = 0.5#np.random.rand()*(u[0]-l[0])+l[0]
        x[1] = 0.5#p.random.rand()*(u[1]-l[1])+l[1]
        return x


    def initial_create(self):
        '''
        Defines and evaluates initial seeding. 
        Saves LHS evaluations to file for quicker model initialisation when testing
        '''
        # Create 2D optimal LHS over [0,1]^n
        sp = samplingplan(2)
        X = sp.optimallhc(30)

        # Stretch LHS over [-1,1]^n
        X = (2*X)-1

        # Evaluates function at LHS points
        y = self.testfunction(X)
        lhs_record = {}
        lhs_record['location'] = copy.deepcopy(X)
        lhs_record['value'] = copy.deepcopy(y)

        with open('init_lhs.p', 'wb') as fp:
            leek.dump(lhs_record, fp, protocol=leek.HIGHEST_PROTOCOL)
    

    def initial_import(self, plot=False):
        '''
        Loads LHS evaluations and initialises the kriging model.
        '''
        with open('init_lhs.p', 'rb') as fp:
            self.lhs_record = leek.load(fp)
        X = self.lhs_record['location']
        y = self.lhs_record['value']
        print(X)
        print(y)

        # Adds the LHS evaluations to other history tracking dictionaries.
        for ii in range(len(y)):
            self.eval_hist['location'].append(list(copy.deepcopy(X[ii])))
            self.eval_hist['value'].append(copy.deepcopy(y[ii]))
            self.krig_hist['location'].append(list(copy.deepcopy(X[ii])))
            self.krig_hist['value'].append(copy.deepcopy(y[ii]))

        # Creates either a kriging or regression kriging model
        print("Setting up Kriging Model")
        if self.regressionswitch == True:
            self.krig = regression_kriging(X, y, name='simple', testPoints=250)
        else:
            self.krig = kriging(X, y, name='simple', testPoints=100)

        # Train model
        self.krig.train(optimizer='ga')
        self.krig.snapshot()

        # Displays initial kriging representation of function.
        if plot == True:
            self.krig.plot()

    
    def latinHypercubeSample(self):
        '''
        Creates a LHS. Currently unused but will be implemented into speeding up the 
        surrogate function search - increase dimentionality.
        '''
        # Initialisation
        n = len(l) 
        PI = np.zeros((n,p))
        for ni in range(n):
            PI[ni,:] = np.arange(1,p+1)
            np.random.shuffle(PI[ni,:])
        xsamp = np.zeros((n,p))
        # Sample construction
        for ii in range(n):
            for jj in range(p):
                r = np.random.randint(-int(1/ds), 1+int(1/ds))
                xsamp[ii,jj] = l[ii] + ((PI[ii,jj]-r)/p) * (u[ii] - l[ii])
        return xsamp


    def rand_vector(self):
        '''
        Returns a random unit vector in R^n using the 2 norm.
        Helper function for the poll direction construction in poll_directions()
        '''
        # Creates a vector spanning all mesh point locations along frame edge
        axis = np.linspace(int(-self.delt/self.ds),int(self.delt/self.ds),int(2*self.delt/self.ds)+1)

        # Selects random value from available axis for each coordinate position
        v = np.zeros(self.n)
        while all(v == np.zeros(self.n)):
            v = np.random.choice(axis,size=self.n)
        
        # Normalises created vector, creating unit vector in R^n
        v = v / (np.linalg.norm(v))

        return v
        
    # Calculates polling directions from a random vector.
    def poll_directions(self):
        '''
        Creates a positive spanning set of poll directions for the poll step.
        '''
        ###1 Create Householder matrix 1###
        # Given v[k] E R^n with ||v[k]||=1 and delt[k] >= ds > 0
        v = self.rand_vector()
        v=v[:,None]
        # Use v[k] to create its associate Houeseholder matrix
        # H[k] = [h1,h2,...,hn]
        H = np.eye(len(v)) - (2*v.dot(v.T))

        ###2 Create poll set 2###
        # Define B[k] = {b1,b2,...,bn} with b[j] = round((delt[k] / ds[k]) * (h[j] / ||h[j]||)) E Z^n
        B = np.zeros((2,len(H.T)))
        for jj in range(len(H[0,:])):
            B[:,jj] = np.round((self.delt/self.ds) * (H[:,jj] / np.linalg.norm(H[:,jj])))
        # Set D[k]_delt = B[k] U (-B[k])
        self.D = B.T.dot(self.Z)


    def iter(self):
        '''
        Updates class at end of iteration, tracking a counter and resetting variables
        '''
        print("Current min at X = "+str(self.X)+" with a value of y = "+str(self.y_min_GLOBAL)+".")
        print("Iter: "+str(self.ii)+". Frame size: "+str(self.delt)+". Mesh size: "+str(self.ds)+".") 
        self.prog_snapshot()
        self.ii=self.ii+1
        self.searchSuccess=False
        self.y = self.y_min_GLOBAL
        
        if self.delt <= self.e:
            self.term = True
        

    def prog_snapshot(self):
        '''
        This function saves a 'snapshot' of the model when the function is called. This allows 
        for a playback of the training process
        '''
        self.history['location'].append(self.X)
        self.history['value'].append(self.y_min_GLOBAL)
        self.history['step_type'].append(copy.deepcopy(self.st))
        self.history['iteration'].append(copy.deepcopy(self.ii))

        with open('eval_hist.p', 'wb') as fp:
            leek.dump(self.eval_hist, fp, protocol=leek.HIGHEST_PROTOCOL)
        with open('krig_hist.p', 'wb') as fp:
            leek.dump(self.krig_hist, fp, protocol=leek.HIGHEST_PROTOCOL)
    
    def import_hist(self, plot=False):
        '''
        Imports saved history files enabling quick restarting in the event of a crash
        or other issue.
        '''
        # Import _hist libraries
        with open('krig_hist.p', 'rb') as fp:
            self.krig_hist = leek.load(fp)
        with open('eval_hist.p', 'rb') as fp:
            self.eval_hist = leek.load(fp)

        X=[]
        y=[]
        for ii in range(len(self.krig_hist['value'])):
            X.append(list(copy.deepcopy(self.krig_hist['location'][ii])))
            if type(self.krig_hist['value'][ii]) == np.ndarray:
                y.append(self.krig_hist['value'][ii][0])
            else:
                y.append(self.krig_hist['value'][ii])
            print(type(self.krig_hist['value'][ii]))

        X = np.array(X)
        y = np.array(y)

        # Creates either a kriging or regression kriging model based on 
        # previous values used.
        print("Setting up Kriging Model")
        if self.regressionswitch == True:
            self.krig = regression_kriging(X, y, name='simple', testPoints=100)
        else:
            self.krig = kriging(X, y, name='simple', testPoints=100)

        # Train model
        self.krig.train(optimizer='ga')
        self.krig.snapshot()

        if plot==True:
            self.krig.plot()


if __name__ == "__main__":
    # Initialise class with regressing and opportunistic polling enabled
    optim = MADS_Optimiser(regswitch=True, opp=True)
    # Create initial seeding
    optim.initial_create()
    # Import initial seeding results
    optim.initial_import(plot=True)
    #optim.import_hist()

    # Run optimisation loop
    while optim.term == False:
        optim.search()
        if optim.searchSuccess == False:
            optim.poll()

        optim.parameter_update()
        optim.iter()

    #print("iteration number: "+str(optim.ii))
    print("r^2 history:"+str(optim.krig.history['rsquared']))

    print('Now plotting final results...')

    optim.krig.plot(show=True)

