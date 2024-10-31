"""
Author: Djazy Faradj
This program is a simple, probably not great example of a linear regression program that takes one X and one Y and determines the optimal
weight "w" that best fits the model. It uses concepts like loss, weight, iterations and rate. I did this just as a guess solution to course im taking
before reading the actual way to do it.
Sample outputs:
----------------------------------
Total iterations: 10233
Expected: X=20, Y=1202342
Predicted: X=20, Y=1202341.9999999998
----------------------------------

----------------------------------
Total iterations: 9
Expected: X=20, Y=1202342
Predicted: X=20, Y=1202026.839473152
----------------------------------
"""

from math import sqrt

def predict(X, w):
    y_hat = X * w
    return y_hat

def cost(y_hat, Y):
    return (y_hat - Y) ** 2


def train(X, Y, rate, iter):
    dcost = 0
    loss1 = 0
    sign = -1
    w = 5
    for i in range (iter):
        y_train = predict(X, w)
        loss2 = cost(y_train, Y)
        dcost = loss2 - loss1
        loss1 = loss2

        if (dcost > 0): 
            sign *= -1
            w += sign*rate*sqrt(loss2)
        if (dcost < 0): w += sign*rate*sqrt(loss2)
    return w
        

def main():
    iterations = 10
    X = 20
    Y = 1202342
    for i in range(1, iterations):
        itr = i
        w = train(X, Y, .03, itr)
        y_hat = predict(20, w)
    print("Total iterations: " + str(itr))
    print ("Expected: X=20, Y=" + str(Y))
    print ("Predicted: X=20, Y=" + str(y_hat))

main()

"""
This is the code borrowed from educative
------------------
def loss(X, Y, w):
  return np.average((predict(X, w) - Y) ** 2)

def train(X, Y, iterations, lr): 
  w = 0
  for i in range(iterations):
    current_loss = loss(X, Y, w)
    print("Iteration %4d => Loss: %.6f" % (i, current_loss))
    
    if loss(X, Y, w + lr) < current_loss: 
      w += lr
    elif loss(X, Y, w - lr) < current_loss: 
      w -= lr
    else:
      return w
      
  raise Exception("Couldn't converge within %d iterations" % iterations)

# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system
w = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f" % w)

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w)))
"""