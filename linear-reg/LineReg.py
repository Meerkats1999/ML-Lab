import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header_list = ["X","Y"]
df = pd.read_csv("Food-Truck-LineReg.csv", names=header_list)
df.head()

class LinearRegression():
  def __init__(self, df):
    self.x = np.array(df["X"])
    self.y = np.array(df["Y"])
    self.n = np.size(self.x)

  def train(self):
    mean_x = np.mean(self.x) 
    mean_y = np.mean(self.y)
    SS_xy = np.sum(self.y*self.x) - self.n*mean_y*mean_x 
    SS_xx = np.sum(self.x*self.x) - self.n*mean_x*mean_x 
    b1 = SS_xy / SS_xx 
    b0 = mean_y - b1*mean_x
    print("b1: "+str(b1)+" b0: "+str(b0))
    self.y_pred = b0 + b1*self.x
  
  def plot_line(self):
    plt.scatter(self.x, self.y, color = "g",s = 30) 
    plt.plot(self.x, self.y_pred, color = "r") 
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.show()
    
model = LinearRegression(df)
model.train()
model.plot_line()