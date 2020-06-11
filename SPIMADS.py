
import numpy as np
import scipy
from scipy.optimize import minimize
import copy
from matplotlib import pyplot as plt
import pylab
from math import inf

import pyKriging
from pyKriging.regressionkrige import regression_kriging
from pyKriging.samplingplan import samplingplan

from pyKriging.krige import kriging

from mpl_toolkits.mplot3d import axes3d
import inspyred
from random import Random
from time import time
from inspyred import ec
import math as m
from matplotlib.patches import Rectangle

try:
    import cPickle as leek
except ImportError:
    import pickle as leek

def optFunct(X):

    try:
        X.shape[1]
    except:
        X = np.array([X])
    
    if X.shape[1] != optim.n:
        print(X)
        raise Exception
        
    
    #print("First Locations"+str(X))
    Xchecked=[]
    Xcopy=[]
    ycopy=[]
    copylist=[]
    for x in X:
        #print(list(x))
        copycheck = False
        c=0
        for prev in optim.eval_hist['location']:
            #print(prev)
            if all(x == prev):
                copycheck = True
                Xcopy.append(list(x))
                ycopy.append(list(optim.eval_hist['value'])[c])
                #print("Copy!")
                copylist.append(True)
                break
            c=c+1
        if copycheck == False:
            Xchecked.append(list(x))
            copylist.append(False)
            #print("notcopy!")
        #print("checked"+str(Xchecked))

    X = np.array(Xchecked)
    #print(X)
    if X.any():
        x0 = X[:,0]
        #print(x0)
        x1 = X[:,1]
        x2 = 1
        #print(len(x1))
        #y = ((x0-0.2242367)**2 + x1**2)+1
        #y = 10*(((1-(x0)*2)**2) + (100*((x1)*6-(((x0)*2)**2))**2))
        y = 4*x2*(1000* (np.sin(4*np.pi*x0)+np.sin(3*np.pi*x1)) + (10000*((x0+0.5)**2 + (x1+0.8)**2)) + (1+(x0+x1+1)**2 * (19-14*x0-3*x0**2-14*x1+6*x0*x1+3*x1**2)) * (30+(2*x0-3*x1)**2 * (18-32*x0+12*x0**2+48*x1-36*x0*x1+27*x1**2)))
        #print(len(y))
    else:
        y=np.array([])
    yfinal=[]
    xfinal=[]
    d=0
    f=0
    #print("yleng"+str(len(y)))
    #print("ycopyleng"+str(len(ycopy)))
    for ii in range(len(y)+len(ycopy)):
        if copylist[ii] == True:
            yfinal.append(ycopy[d])
            xfinal.append(Xcopy[d])
            d=d+1
        elif copylist[ii] == False:
            yfinal.append(y[f])
            xfinal.append(X[f,:])
            f=f+1
    y=np.array(yfinal)
    X = np.array(xfinal)
    #print("solution="+str(y))
    #print("Final Locations:"+str(X))
    return y

class madsness():
    def __init__(self, dim=2 ,e=1/64, testfunction=optFunct, name='', testPoints=None, delt=1, tau=4, regswitch=False, opp=False, **kwargs):
        self.regressionswitch = regswitch
        self.n = copy.deepcopy(dim)
        self.y = inf
        self.y_min_GLOBAL = inf
        self.delt = copy.deepcopy(delt)         # Frame size parameter (0,inf)
        self.ds = copy.deepcopy(delt)           # Mesh size parameter (0,inf)
        self.tau = copy.deepcopy(tau)           # Mesh size adjustment parameter (0,1; rational)
        self.G = np.eye(self.n)                      # Invertiuble matrix defining mesh 'shape'.
        z1 = np.eye(self.n)
        z2 = -np.ones((self.n,1))
        self.Z = np.append(z1,z2,axis=1)        # Positive spanning set for R^n
        self.D = self.G.dot(self.Z)             # Set of directions in R^n
        self.e = copy.deepcopy(e)               # Mesh size limit

        self.testfunction = testfunction    
        self.name = name

        self.opp = opp

        self.ii = 0
        self.term = False

        self.history = {}
        self.history['location'] = []
        self.history['value'] = []
        self.history['step_type'] = []
        self.history['iteration'] = []

        self.eval_hist = {}
        self.eval_hist['location'] = []
        self.eval_hist['value'] = []

        self.krig_hist = {}
        self.krig_hist['location'] = []
        self.krig_hist['value'] = []

        self.lhs_record = {}
        self.lhs_record['location'] = []
        self.lhs_record['value'] = []

    # Main Functions are:
    #     parameter_update()
    #     search()
    #     poll()

    # Finds mesh size based on frame size
    def parameter_update(self):
        self.ds = min([self.delt,self.delt**2])

    # Search the parameter space on grid points, until a certain mesh refinement is reached. 
    # At this point a LHS method is used to select random grid points.
    def search(self):
        #try searching all grid points (for now)
        Mk = np.linspace(-1, 1, min([int(2/self.ds)+1,int(2*32)+1]))

        xv, yv, zv = np.meshgrid(Mk,Mk,Mk)

        xe = np.append(xv,yv)
        xe = np.append(xe,zv)
        xe = np.reshape(xe, (self.n,len(Mk)**self.n))
        print(xe)
        # Evaluate surrogate function at all points on Mk
        ye = np.array([])
        for x in xe.T:
            print(x)
            ye=np.append(ye,self.krig.predict(x))

        yxe=sorted(zip(ye,xe.T))

        print(yxe)

        # Find the '3' lowest estimates
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
                if (np.linalg.norm(pointdist1) > 0.5) & (np.linalg.norm(pointdist2) > 0.5):
                    threesmall.append(yxe[c])
            c=c+1

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

        # Check if a new global minimum has been found
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

        # Add simulated points to kriging model
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

    # Searches points around the current global minimum, looking for incremental improvements
    def poll(self):
        # Find set of n+1 poll points based on random poll_directions function
        self.poll_directions()
        self.P = (self.D*self.ds) + np.repeat(self.X, self.D.shape[1]).reshape(self.Z.shape)
        print(self.P)
        print(self.X)

        plt.plot(self.X[0],self.X[1],'+')
        plt.plot(self.P[0,:],self.P[1,:],'x')
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((self.X[0] - self.delt, self.X[1] - self.delt), 2*self.delt, 2*self.delt, facecolor="grey"))
        plt.show()

        if self.opp == True:
            pollsucc = False
            yPolMin = inf
            for xPol in self.P.T:
                print("Poll Point: "+str(xPol))

                iscopy=False
                for cc in range(len(self.eval_hist['location'])):
                    if np.all(xPol == self.eval_hist['location'][cc]):
                        iscopy = True
                        print("copy")
                        yPol = self.eval_hist['value'][cc]
                        break
                if (iscopy == False):
                    yPol = self.testfunction(xPol)
                
                self.eval_hist['location'].append(list(copy.deepcopy(xPol)))
                self.eval_hist['value'].append(copy.deepcopy(yPol))
                # Barrier function - sets points outside of constraint to inf
                print(yPol)
                if any(abs(xPol) > 1):
                    yPol = inf

                if yPol < yPolMin:
                    yPolMin = yPol
                print(yPol)

                # Check if a new global minimum has been found
                if yPolMin < self.y_min_GLOBAL:
                    self.X = copy.deepcopy(xPol)
                    self.delt = min([self.delt * self.tau,1])
                    self.y_min_GLOBAL = yPolMin
                    lineSearch = True
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
            # Evaluates  poll points using funtion
            
            
            yPol = self.testfunction(self.P.T)

            

            yPol=np.array(yPol)
            print(self.P.T)
            print(yPol)

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
                lineSearch = True
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
        
    # Sets term variable to True.
    def termination(self):
        self.term==True

    # Defines a starting point within a range.
    def startPoint(self, l, u):
        n = len(l)
        x = np.zeros(n)
        x[0] = 0.5#np.random.rand()*(u[0]-l[0])+l[0]
        x[1] = 0.5#p.random.rand()*(u[1]-l[1])+l[1]
        return x

    # Defines initial seeding and initialises kriging model.
    def initial_create(self):
        # Create 2D optimal LHS over [0,1]^n
        sp = samplingplan(self.n)
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
    
    def initial_import(self):

        with open('init_lhs.p', 'rb') as fp:
            self.lhs_record = leek.load(fp)

        X = self.lhs_record['location']
        #print("X")
        #print(len(X))
        y = self.lhs_record['value']
        print(X)
        print(y)
        #print("y")
        #print(len(y))
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


    # Creates a LHS based on the current mesh size parameter.
    def latinHypercubeSample(self):
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

    # Returns a random unit vector in R^n using the 2 norm
    def rand_vector(self):
        # Creates a vector spanning all mesh point locations along frame edge
        axis = np.linspace(int(-self.delt/self.ds),int(self.delt/self.ds),int(2*self.delt/self.ds)+1)

        # Selects random value from available axis for each coordinate position
        v = np.zeros(self.n)
        while all(v == np.zeros(self.n)):
            v = np.random.choice(axis,size=self.n)
        
        # Normalises created vector, creating unit vector in R^n
        v = v / (np.linalg.norm(v))
        #print(v)
        return v
        
    # Calculates polling directions from a random vector.
    def poll_directions(self):
        ###1 Create Householder matrix 1###
        # Given v[k] E R^n with ||v[k]||=1 and delt[k] >= ds > 0
        v = self.rand_vector()
        v=v[:,None]
        # Use v[k] to create its associate Houeseholder matrix
        # H[k] = [h1,h2,...,hn]
        H = np.eye(len(v)) - (2*v.dot(v.T))

        ###2 Create poll set 2###
        # Define B[k] = {b1,b2,...,bn} with b[j] = round((delt[k] / ds[k]) * (h[j] / ||h[j]||)) E Z^n
        B = np.zeros_like(H.T)
        for jj in range(len(H[0,:])):
            B[:,jj] = np.round((self.delt/self.ds) * (H[:,jj] / np.linalg.norm(H[:,jj])))
        # Set D[k]_delt = B[k] U (-B[k])
        self.D = B.T.dot(self.Z)

    # Updates class at end of iteration, tracking a counter and resetting variables
    def iter(self):
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
        This function saves a 'snapshot' of the model when the function is called. This allows for a playback of the training process
        '''
        self.history['location'].append(self.X)
        self.history['value'].append(self.y_min_GLOBAL)
        self.history['step_type'].append(copy.deepcopy(self.st))
        self.history['iteration'].append(copy.deepcopy(self.ii))

        with open('eval_hist.p', 'wb') as fp:
            leek.dump(self.eval_hist, fp, protocol=leek.HIGHEST_PROTOCOL)
        with open('krig_hist.p', 'wb') as fp:
            leek.dump(self.krig_hist, fp, protocol=leek.HIGHEST_PROTOCOL)
    
    def import_hist(self):
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
        # Creates either a kriging or regression kriging model
        print("Setting up Kriging Model")
        if self.regressionswitch == True:
            self.krig = regression_kriging(X, y, name='simple', testPoints=100)
        else:
            self.krig = kriging(X, y, name='simple', testPoints=100)
        # Train model
        self.krig.train(optimizer='ga')
        self.krig.snapshot()

if __name__ == "__main__":
    optim = madsness(dim=3, regswitch=True, opp=True)
    #optim.initial_create()
    optim.initial_import()
    #optim.import_hist()

    #optim.krig.plot()

    while optim.term == False:
        optim.search()
        if optim.searchSuccess == False:
            optim.poll()

        optim.parameter_update()
        #print(optim.delt)
        optim.iter()
    #print("iteration number: "+str(optim.ii))
    print("r^2 history:"+str(optim.krig.history['rsquared']))

    print('Now plotting final results...')

    optim.krig.plot(show=True)
