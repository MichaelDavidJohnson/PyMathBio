
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


        
        
