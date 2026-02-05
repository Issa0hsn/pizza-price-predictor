import numpy as np
import matplotlib.pyplot as plt

# X_train: {pizza size, peperoni slices, cheese price}
x_train = np.array([
    [10.0, 0, 2.5], [12.0, 4, 3.0], [14.0, 6, 2.8], [16.0, 8, 4.5], [18.0, 10, 4.0],
    [20.0, 12, 5.0], [22.0, 14, 5.5], [24.0, 16, 6.0], [26.0, 18, 5.8], [28.0, 20, 7.0],
    [30.0, 22, 7.5], [32.0, 24, 8.0], [34.0, 26, 8.2], [36.0, 28, 9.0], [38.0, 30, 9.5],
    [40.0, 30, 10.0], [11.0, 2, 2.2], [13.0, 5, 3.5], [15.0, 7, 3.1], [17.0, 9, 4.2],
    [19.0, 11, 4.8], [21.0, 13, 5.2], [23.0, 15, 6.1], [25.0, 17, 6.5], [27.0, 19, 6.9],
    [29.0, 21, 7.2], [31.0, 23, 7.8], [33.0, 25, 8.5], [35.0, 27, 9.2], [37.0, 29, 9.8],
    [10.5, 1, 2.4], [12.5, 3, 3.2], [14.5, 5, 2.9], [16.5, 7, 4.4], [18.5, 9, 4.1],
    [20.5, 11, 5.1], [22.5, 13, 5.6], [24.5, 15, 6.2], [26.5, 17, 5.9], [28.5, 19, 7.1],
    [30.5, 21, 7.6], [32.5, 23, 8.1], [34.5, 25, 8.3], [36.5, 27, 9.1], [38.5, 29, 9.6],
    [15.0, 10, 5.0], [25.0, 5, 3.0], [35.0, 15, 7.0], [20.0, 20, 6.0], [30.0, 10, 8.0]
])

# y_train: pizza price 
y_train = np.array([
    12.5, 15.2, 18.1, 22.4, 25.0, 29.3, 33.1, 37.5, 40.2, 45.8,
    49.5, 54.0, 58.2, 63.5, 68.1, 72.0, 13.1, 16.8, 19.5, 23.8,
    27.5, 31.2, 35.8, 40.5, 44.1, 48.2, 53.1, 57.5, 62.8, 67.5,
    12.8, 15.9, 18.6, 22.9, 25.6, 30.1, 33.8, 38.2, 41.0, 46.5,
    50.2, 54.8, 58.9, 64.2, 68.9, 23.5, 33.0, 55.0, 35.5, 48.0
])
n=x_train.shape[0]
x_train
def z_score(x,n):
    if n==1:
        m=1
    else:    
        m=np.mean(x)
    
    s = ((np.sum((x-m)**2))/n)**(1/2)
    return (x-m)/s
def rescale(x1,x2,n):
    m=np.mean(x2)
    s = ((np.sum((x2-m)**2))/n)**(1/2)
    return x1*s + m
  
scaledx1=z_score(x_train[:,0],n)
scaledx2=z_score(x_train[:,1],n)
scaledx3=z_score(x_train[:,2],n)
scaled_x=np.copy(x_train)
scaled_x[:,0]=scaledx1
scaled_x[:,1]=scaledx2
scaled_x[:,2]=scaledx3

w=np.array([1.,1.,1.])
b=3
def pre_func(x,w,b):
    return np.dot(x,w) + b
#print(pre_func(w,b,x_train[0]))
def cost_func(w,b,y,x,n):
    return np.dot((pre_func(x,w,b)-y),(pre_func(x,w,b)-y).T)/(2*n)


history_cost=[] 
iteration=[0]
def grad(w,b,x,y,n,a):
    dw=np.copy(w)
    db=b-a*np.sum((pre_func(x,w,b)-y)/n)
    
    dw=np.dot(x.T,(pre_func(x,w,b)-y))/n
    return w-a*dw,db
#print(grad(w,b,scaled_x,y_train,n,0.1))
alpha=0.3
scale=10
  
history_cost.append(cost_func(w,b,y_train,scaled_x,n))
for i in range(10000):
    w,b=grad(w,b,scaled_x,y_train,n,alpha)
    if i%200==0:
        history_cost.append(cost_func(w,b,y_train,scaled_x,n))
        iteration.append(i)
    
  

plt.scatter(pre_func(scaled_x,w,b),y_train, label='خط', color='red', linewidth=2)
plt.xlabel('prediction')
plt.ylabel("real price")
plt.grid(True)
#plt.legend()


final_predictions = pre_func(scaled_x,w,b)

fig = plt.figure(figsize=(12, 10))

ax1 = plt.subplot(2, 1, 1) 
ax1.plot(history_cost)
ax1.set_title("Learning Curve: Cost vs. Iterations")
ax1.set_ylabel("Cost (J)")
ax1.set_xlabel("Iterations (x100 if sampled)")
ax1.grid(True)

feature_names = ["Size (Inch)", "Pepperoni Count", "Cheese Price ($)"]

for i in range(3):
    ax = plt.subplot(2, 3, i + 4) 
    
    ax.scatter(x_train[:, i], y_train, color='blue', label='Actual', alpha=0.6)
    
    ax.scatter(x_train[:, i], final_predictions, color='red', label='Predicted', marker='x')
    
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout() 
plt.show()

c=input("would you like me  to predicte how much your pizza cost  ?(y/n)\n")
if c.lower()=='y':
    s=z_score(int(input("Enter size(by square inch ): ")),1)
    p=z_score(int(input("Enter number of pepperoni slices : ")),1)
    d=z_score(int(input("Enter cheese price (by dollar): ")),1)
    x=[s,p,d]

    print(f"Your pizza will cost about {pre_func(x,w,b):.2f}")

