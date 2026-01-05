import pandas as pd 
import matplotlib.pyplot as plt 

data=pd.read_csv('studytime.csv')

def loss_function(m,b,points):
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i].studytime
        y=points.iloc[i].score
        total_error+=(y-(m*x+b))**2
    return total_error/float(len(points))

def gradient_descent(m_now,b_now,points,L):
    m_gradient=0
    b_gradient=0

    n=len(points)

    for i in range(n):
        x=points.iloc[i].studytime
        y=points.iloc[i].score

        m_gradient += -(2/n) * x * (y-(m_now * x + b_now))
        b_gradient += -(2/n) * (y-(m_now * x + b_now))
    
    m = m_now - m_gradient * L 
    b = b_now - b_gradient * L 
    return m,b 

m=0 
b=0
L=0.0001
epochs=2000



# Calculating the gradient descent 

for i in range(epochs):
    m,b = gradient_descent(m,b,data,L)
print ("Final m and b: ", m,b)

#Plotting regression line


x_line = [data.studytime.min(),data.studytime.max()]
y_line = [m * x_line[0] + b,m * x_line[1] + b]

plt.scatter(data.studytime,data.score,color="black")
plt.plot(x_line,y_line,color="red",linewidth=2)
plt.xlabel("Study Time")
plt.ylabel("Score")
plt.title("Linear Regression from Scratch")
plt.show