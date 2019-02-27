
class PhasePlane2:
    def __init__(self,f,g):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.integrate import solve_ivp
        self.solve_ivp = solve_ivp
        self.f = f
        self.g = g
        #self.iterations = iterations
        #self.initialX = initialX
        #self.initialY = initialY
        self.xMin = 0
        self.xMax = 1
        self.yMin = 0
        self.yMax = 1
        self.noArrowX = 10
        self.noArrowY = 10
        self.arrows = True
        self.steadyStates = True
        self.trajectories = False
    def getFlow(self,initialX,initialY,iterations = 100,minT=0,maxT=100):    
        def dydt(t,y):
            y1,y2 = y
            dy1dt = self.f(y1,y2)
            dy2dt = self.g(y1,y2)
            return [dy1dt,dy2dt]
        sol = self.solve_ivp(dydt,[minT,maxT],[initialX,initialY])
        return sol
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
    def findSteadyStates(self,epsilon):
        return 0
    def draw(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.axis("scaled")
        ax1.axis([self.xMin,self.xMax,self.yMin,self.yMax])
        if self.arrows == True:
            dx = (self.xMax-self.xMin) / (self.noArrowX )
            dy = (self.yMax-self.yMin) / (self.noArrowY )
            for j in range(self.noArrowY):
                y = self.yMin + (j + 0.5)*dx
                for i in range(self.noArrowX):
                    x = self.xMin + (i + 0.5)*dx
                    u = self.f(x,y)
                    v = self.g(x,y)
                    l = np.sqrt(np.square(u)+np.square(v))
                    x0 = x - (0.3*np.min([dx,dy])*u) / l
                    y0 = y - (0.3*np.min([dx,dy])*v) / l
                    delx =  (0.6*np.min([dx,dy])*u) / l
                    dely =  (0.6*np.min([dx,dy])*v) / l
                    ax1.arrow(x0,y0,delx,dely,head_width = 0.2*np.min([dx,dy]) )
        if self.trajectories == True:
            for n in range(self.nTrajectories):
                sol = self.getFlow(self.initialXs[n],self.initialYs[n])
                ax1.plot(sol.y[0],sol.y[1])
                print(sol.y)
            return fig.show()


        
    
    
    