import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.misc import derivative
class twoDSystem:

    def __init__(self,f,g):
        
        self.solve_ivp = solve_ivp
        self.f = f
        self.g = g
        self.xMin = 0
        self.xMax = 1
        self.yMin = 0
        self.yMax = 1
        self.noArrowX = 10
        self.noArrowY = 10
        self.arrows = True
        self.steadyStates = False
        self.trajectories = False
        self.title = "Title"
		self.numericalSteadyStates = False
    
    def setArrows(ArrowNo = 10):
        self.noArrowX = ArrowNo
        self.noArrowY = ArrowNo
    
    
    def setTrajectories(self,initialXs,initialYs):
        self.initialXs = np.array(initialXs)
        self.initialYs = np.array(initialYs)
        tX = self.initialXs.size
        tY = self.initialYs.size
        if tX == tY:
            self.trajectories = True 
            self.nTrajectories = tX
            return self.initialXs,self.initialYs
        else:
            self.trajectories = False 
            print("Initial condition array sizes don't agree!")
    
    
    def setPlotRange(self,xMin = 0. ,xMax = 1.,yMin = 0.,yMax = 1.):
        self.xMin = xMin
        self.xMax = xMax 
        self.yMin = yMin
        self.yMax = yMax


    def setPlotTitle(self,title):
        self.title = title
        try:
            if not type(title) == str:
                raise ValueError
        except ValueError:
            self.title = "Title"
            print("Error in setPlotTitle : Please input a string")


    def setSteadyStates(self,steadyStateX,steadyStateY,colour = "r"):
        self.steadyStateX = (steadyStateX)
        self.steadyStateY = (steadyStateY)
        if colour == "r" or colour == "b" or colour == "g" or colour == "c" or colour == "m" or colour == "y" or colour == "b" or colour == "w":
			self.steadyStateColour = colour
		else:
			print("Invalid colour choice, see .plotoptions() for help.")
        self.steadyStates = True
        
    def setDiffs(self,fx,fy,gx,gy):
        self.fx = fx
        self.fy = fy
        self.gx = gx
        self.gy = gy
    def getJacobian(self,x0,y0):
        jOut = np.array([ [self.fx(x=x0,y=y0),self.fy(x=x0,y=y0)],[self.gx(x=x0,y=y0),self.gy(x=x0,y=y0)]])
        return jOut
    def getIJacobian(self,x0,y0):
        det = self.fx(x=x0,y=y0)*self.gy(x=x0,y=y0) - self.fy(x=x0,y=y0)*self.gx(x=x0,y=y0)
        try:
            if det==0:
                raise ValueError
        except ValueError:
                print("Determinant is zero!")
        iOut = (1/det)*np.array([ [self.gy(x=x0,y=y0),-self.fy(x=x0,y=y0)],[-self.gx(x=x0,y=y0),self.fx(x=x0,y=y0)]])
        return iOut
        
        
    
    def getFlow(self,initialX,initialY,iterations = 100,minT=0,maxT=100):    
        def dydt(t,y):
            y1,y2 = y
            dy1dt = self.f(y1,y2)
            dy2dt = self.g(y1,y2)
            return [dy1dt,dy2dt]
        sol = self.solve_ivp(dydt,[minT,maxT],[initialX,initialY])
		
        return sol
    
	def setFlowColour(self,colour = "g"):
        if colour == "r" or colour == "b" or colour == "g" or colour == "c" or colour == "m" or colour == "y" or colour == "b" or colour == "w":
			self.flowColour = colour
		else:
			print("Invalid colour choice, see .plotoptions() for help.")
	
    def newtonRaphson(self,testX,testY,accuracy = 0.00001):
        self.testX = testX
        self.testY = testY
        def partial_derivative(func, var=0, point=[]):
            args = point[:]
            def wraps(x):
                args[var] = x
                return func(*args)
            return derivative(wraps, point[var], dx = 1e-10)
        eps = np.sqrt( (self.f(x = self.testX, y = self.testY))**2 + (self.g(x = self.testX, y = self.testY))**2 )
        
        while eps > accuracy:
            functions = np.array([[self.f(x=self.testX,y=self.testY)],[self.g(x=self.testX,y=self.testY)]])
            Jacobian = np.array([[partial_derivative(f,0,[self.testX,self.testY]),partial_derivative(f,1,[self.testX,self.testY])],
                                [partial_derivative(g,0,[self.testX,self.testY]),partial_derivative(g,1,[self.testX,self.testY])]])
            jacobianInverse = np.linalg.inv(Jacobian)
            tempArray = np.matmul(jacobianInverse, functions)
            self.testX = self.testX - tempArray[0][0]
            self.testY = self.testY - tempArray[1][0]
            eps = np.sqrt( (self.f(x = self.testX, y = self.testY))**2 + (self.g(x = self.testX, y = self.testY))**2 )
        print("There is a steady state at ", [self.testX,self.testY])
		
        return self.testX,self.testY
    
    def newtonRaphsonAnalytic(self,testX,testY,accuracy = 0.00001):
        tX = testX
        tY = testY
        eps = np.sqrt( (self.f(x = tX, y = tY))**2 + (self.g(x = tX, y = tY))**2 )
        while eps > accuracy:
            functions = np.array([[self.f(x=tX,y=tY)],[self.g(x=tX,y=tY)]])
            jacobianInverse = self.getIJacobian(tX,tY)
            tempArray = np.matmul(jacobianInverse,functions)
            tX -=  tempArray[0][0]
            tY -=  tempArray[1][0]
            eps = np.sqrt( (self.f(x = tX, y = tY))**2 + (self.g(x = tX, y = tY))**2 )
        print("There is a steady state at ", [tX,tY])
		
        return tX,tY

    def findSteadyStates(self,epsilon = 0.01,method = "Exhaustive"):
        if method == "Exhaustive":
            self.steadyStateArray = []
            def frange(x, y, jump):
                while x < y:
                    yield x
                    x += jump
            for y in frange(self.yMin,self.yMax,epsilon):
                for x in frange(self.xMin,self.xMax,epsilon):
                    tempX,tempY = self.newtonRaphson(x,y,0.0000001)
                    
                    isInCell = (tempX > x - 0.5*epsilon) and (tempX <= x + 0.5*epsilon) and (tempY > y - 0.5*epsilon) and (tempY <= y + 0.5*epsilon)
                                    
                    
                    if isInCell:
                        self.steadyStateArray.append([tempX,tempY])
                    

            print(self.steadyStateArray)
            self.numericalSteadyStates = True
			
        if method == "Analytic":
            self.steadyStateXArray = []
			def frange(x,y,jump):
				while x<y:
					yield x
					x += jump
			for y in frange(self.yMin,self.yMax,epsilon):
				for x in frange(self.xMin,self.xMax,epsilon):
					tempX,tempY = self.newtonRaphsonAnalytic(x,y)
					isInCell = (tempX > x - 0.5*epsilon) and (tempX <= x + 0.5*epsilon) and (tempY > y - 0.5*epsilon) and (tempY <= y + 0.5*epsilon)
					if isInCell:
                        self.steadyStateArray.append([tempX,tempY])
       
    
    def getStability(self):
        
        if self.numericalSteadyStates == False:
			def partial_derivative(func, var=0, point=[]):
				args = point[:]
				def wraps(x):
					args[var] = x
					return func(*args)
				return derivative(wraps, point[var], dx = 1e-14)
        
			for n in range(len(self.steadyStateX)):
				Jacobfx = partial_derivative(f,0,[self.steadyStateX[n],self.steadyStateY[n]])
				Jacobgx = partial_derivative(g,0,[self.steadyStateX[n],self.steadyStateY[n]])
				Jacobfy = partial_derivative(f,1,[self.steadyStateX[n],self.steadyStateY[n]])
				Jacobgy = partial_derivative(g,1,[self.steadyStateX[n],self.steadyStateY[n]])
				trace = Jacobfx + Jacobgy
				det = ( Jacobfx * Jacobgy ) - ( Jacobfy * Jacobgx )
				disc = trace**2 - 4*det
				if det < 0:
					print("The Steady State at ",(self.steadyStateX[n],self.steadyStateY[n]),"is a Saddle Node")
				elif det > 0 and trace == 0:
					print("The Steady State at ",(self.steadyStateX[n],self.steadyStateY[n])," is a Centre")
				elif det > 0 and trace > 0 and disc > 0:
					print("The Steady State at ",(self.steadyStateX[n],self.steadyStateY[n])," is an Unstable Node")
				elif det > 0 and trace < 0 and disc > 0:
					print("The Steady State at ",(self.steadyStateX[n],self.steadyStateY[n])," is a Stable Node")
				elif det > 0 and trace > 0 and disc < 0:
					print("The Steady State at ",(self.steadyStateX[n],self.steadyStateY[n]), " is an Unstable Focus")
				elif det > 0 and trace < 0 and disc < 0:
					print("The Steady State at ",(self.steadyStateX[n],self.steadyStateY[n])," is a Stable Focus")
        elif self.numericalSteadyStates == True:
			def partial_derivative(func,var = 0,point = []):
				args = point[:]
				def wraps(x):
					args[var] = x
					return funct(*args)
				return derivative(wraps,point[var],dx = 1e-14
			for n in range(len(self.steadyStateArray)):
				Jacobfx = partial_derivative(f,0,[self.steadyStateArray[n,0],self.steadyStateArray[n,1]])
				Jacobgx = partial_derivative(g,0,[self.steadyStateArray[n,0],self.steadyStateArray[n,1]])
				Jacobfy = partial_derivative(f,1,[self.steadyStateArray[n,0],self.steadyStateArray[n,1]])
				Jacobgy = partial_derivative(g,1,[self.steadyStateArray[n,0],self.steadyStateArray[n,1]])
				trace = Jacobfx + Jacobgy
				det = ( Jacobfx * Jacobgy ) - ( Jacobfy * Jacobgx )
				disc = trace**2 - 4*det			
				if det < 0:
					print("The Steady State at ",(self.steadyStateArray[n,0],self.steadyStateArray[n,1]),"is a Saddle Node")
				elif det > 0 and trace == 0:
					print("The Steady State at ",(self.steadyStateArray[n,0],self.steadyStateArray[n,1])," is a Centre")
				elif det > 0 and trace > 0 and disc > 0:
					print("The Steady State at ",(self.steadyStateArray[n,0],self.steadyStateArray[n,1])," is an Unstable Node")
				elif det > 0 and trace < 0 and disc > 0:
					print("The Steady State at ",(self.steadyStateArray[n,0],self.steadyStateArray[n,1])," is a Stable Node")
				elif det > 0 and trace > 0 and disc < 0:
					print("The Steady State at ",(self.steadyStateArray[n,0],self.steadyStateArray[n,1]), " is an Unstable Focus")
				elif det > 0 and trace < 0 and disc < 0:
					print("The Steady State at ",(self.steadyStateArray[n,0],self.steadyStateArray[n,1])," is a Stable Focus")    

    def setNullClines(self):
        return 0
    
    def plotoptions(self):
		print("Plot Colours:")
		print("Red = "r"")
		print("Blue = "b"")
		print("Green = "g"")
		print("Cyan = "c"")
		print("Magenta = "m"")
		print("Yellow = "y"")
		print("Black = "k"")
		print("White = "w"")
	
	def setDrawTime(self):
        return 0
    
    def draw(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.axis("scaled")
        ax1.axis([self.xMin,self.xMax,self.yMin,self.yMax])
        ax1.set_title(self.title)
        if self.arrows == True:
            offset = 0.3
            scale = 0.6
            dx = (self.xMax-self.xMin) / (self.noArrowX )
            dy = (self.yMax-self.yMin) / (self.noArrowY )
            headWidth = 0.2*np.min([dx,dy])
            for j in range(self.noArrowY):
                y = self.yMin + (j + 0.5)*dx
                for i in range(self.noArrowX):
                    x = self.xMin + (i + 0.5)*dx
                    u = self.f(x,y)
                    v = self.g(x,y)
                    l = np.sqrt(np.square(u)+np.square(v))
                    x0 = x - (offset*np.min([dx,dy])*u) / l
                    y0 = y - (offset*np.min([dx,dy])*v) / l
                    delx =  (scale*np.min([dx,dy])*u) / l
                    dely =  (scale*np.min([dx,dy])*v) / l
                    ax1.arrow(x0,y0,delx,dely,head_width = headWidth  )
        if self.trajectories == True:
            for n in range(self.nTrajectories):
                sol = self.getFlow(self.initialXs[n],self.initialYs[n])
                ax1.plot(sol.y[0],sol.y[1],self.flowColour)
                
        if self.steadyStates == True:
            for n in range(len(self.steadyStateX)):
                ax1.plot(self.steadyStateX[n],self.steadyStateY[n],'o',color = self.steadyStateColour)
		
		if self.numericalSteadyStates == True:
			for n in range(len(self.steadyStateArray)):
				ax1.plot(self.steadyStateArray[n,0],self.steadyStateArray[n,1],'o',color = self.steadyStateColour)
				
		
        return fig.show()


        
        
    


        
        
    
    
    
    
class Logistic:
    def __init__(self,r):
        self.r = r
    def function(self,xMin = 0 ,xMax = 1,iterations = 100):
        self.xMin = xMin
        self.xMax = xMax
        self.iterations = iterations
        self.storage = []
        self.xrange = np.linspace(self.xMin,self.xMax,self.iterations)
        for i in self.xrange:
            temp = self.r*i*(1-i)
            self.storage.append(temp)
        return(self.storage)
    def draw(self,title = "Logistic Map",xAxis = "x",yAxis = "y",yMin = 0,yMax = 1):
        self.title = str(title)
        self.xAxis = str(xAxis)
        self.yAxis = str(yAxis)
        self.yMin = yMin
        self.yMax = yMax
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.axis("scaled")
        ax1.axis([self.xMin,self.xMax,self.yMin,self.yMax])
        ax1.plot(self.xrange,self.function())
        return fig.show()


        
        
    
    
    
