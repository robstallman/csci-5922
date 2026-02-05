#%% Imports
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#%% Function definitions
def cross_entropy_loss(y_hat, y):
    y_hat = np.float64(y_hat)
    y = np.float64(y)

    # Cross-entropy loss
    return -np.sum(y * np.log(y_hat))

def binary_cross_entropy_loss(y_hat: float, y: float):
    return -1*(y*np.log(y_hat) + (1-y)*np.log((1-y_hat)))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(x):
    x = x.flatten()
    y = np.array([np.e ** x_i for x_i in x])
    return y / np.sum(y)

# %%
##############
# Question 2 #
##############

#%% Part (a)
X1 = 1
X2 = -1
W1 = -0.5
W2 = 1.0
Y = 0
H1 = X1*X2
H2 = min(X1, H1)
H3 = W1*H1 - W2*H2
Y_hat = sigmoid(H3 + H2)
L = binary_cross_entropy_loss(Y_hat, Y)
dLdy = (Y_hat-Y)/(Y_hat*(1-Y_hat))
dLdH2 = dLdy * sigmoid_prime(H2)
dLdH3 = dLdy * sigmoid_prime(H3)
dLdW1 = dLdH3 * H1
dLdW2 = dLdH3 * H2 * -1
dH2dH1 = 1 if H1 < X1 else 0
dH2dX1 = 0 if H1 < X1 else 1
dLdH1 = dLdH3*W1 + dLdH2*dH2dH1
dLdX1 = dLdH1*X2 + dLdH2*dH2dH1
dLdX2 = dLdH1*X1

print(f"H1 = {H1}")
print(f"H2 = {H2}")
print(f"H3 = {H3}")
print(f"Y_hat = {Y_hat}")
print(f"L = {L}")
# %% Part (b)
X = np.array([1.0,-1.0,0.5])
W1 = np.array([-0.5,0.0,0.5])
W2 = np.array([[1,2,-1],[-1,0,2],[0,-2,1]])
x = np.convolve(W1,X,mode='same')
H1 = (np.maximum(x, 0.1*x) + x).reshape(3,-1)
H2 = np.matmul(W2, H1)
y_hat = softmax(H2).reshape(3,-1)
y = np.array([[0],[1],[0]])
L = cross_entropy_loss(y_hat, y)
dLdH2 = (y_hat - y)
dLdW2 = np.matmul(dLdH2, H1.reshape(1,-1))
dLdH1 = np.matmul(W2.transpose(), dLdH2)

# %%
