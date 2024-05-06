# Source: https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/mTSP_en.ipynb#scrollTo=3CNaCI7VhE_x

################################################
# Loading libraries
################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from geopy import distance # Library for geographical calculations

################################################
# Original Data
################################################
def generate_random_points(num_points, min_lat, max_lat, min_lon, max_lon):
    random_points = []
    for _ in range(num_points):
        lat = np.round(np.random.uniform(min_lat, max_lat), 4)
        lon = np.round(np.random.uniform(min_lon, max_lon), 4)
        random_points.append((lat, lon))
    return random_points

# Exemple d'utilisation
num_points = 20
min_lat = -12.1
max_lat = -11.9
min_lon = -77.1
max_lon = -76.9

#points = generate_random_points(num_points, min_lat, max_lat, min_lon, max_lon)
points = [(-12.059296, -76.975893),
          (-12.079575, -77.009686),
          (-12.087303, -76.996620),
          (-12.084391, -76.975651),
          (-12.063603, -76.960483),
          (-12.056762, -77.014452),
          (-12.011531, -77.002383),
          (-12.011531, -77.002353),
          (-12.011331, -77.002383),
          (-12.014531, -77.002323),
          (-12.011531, -77.102383),
          (-12.013531, -77.002383),
          (-12.013531, -77.004383),
          (-12.021531, -77.002343),
          (-12.045531, -77.002323),
          ]
################################################
# Building distance matrix
################################################
n = len(points)
C = np.zeros((n,n))

for i in range(0, n):
    for j in range(0, len(points)):
        C[i,j] = distance.distance(points[i], points[j]).km

# Showing distance matrix
print('Distance Matrix is:\n')  
print(np.round(C,4))

################################################
# Plotting the optimal path
################################################

# Transforming the points to the xy plane approximately
xy_cords = np.zeros((n,2))

for i in range(0, n):
    xy_cords[i,0] = distance.distance((points[0][1],0), (points[i][1],0)).km
    xy_cords[i,1] = distance.distance((0,points[0][0]), (0,points[i][0])).km

################################################
# Solving the integer programming problem
################################################

# Defining the variables
X = cp.Variable(C.shape, boolean=True)
u = cp.Variable(n, integer=True)
m = 3
ones = np.ones((n,1))

# Defining the objective function
objective = cp.Minimize(cp.sum(cp.multiply(C, X)))

# Defining the constraints
constraints = []
constraints += [X[0,:] @ ones == m]
constraints += [X[:,0] @ ones == m]
constraints += [X[1:,:] @ ones == 1]
constraints += [X[:,1:].T @ ones == 1]
constraints += [cp.diag(X) == 0]
constraints += [u[1:] >= 2]
constraints += [u[1:] <= n]
constraints += [u[0] == 1]

for i in range(1, n):
    for j in range(1, n):
        if i != j:
            constraints += [ u[i] - u[j] + 1  <= (n - 1) * (1 - X[i, j]) ]

# Solving the problem
prob = cp.Problem(objective, constraints)
prob.solve(verbose=False)

# Transforming the solution to paths
X_sol = np.argwhere(X.value==1)

ruta = {}
print(X_sol)
for i in range(0, m):
    ruta['Salesman_' + str(i+1)] = [0]
    j = i
    a = 10e10
    while a != 0:
        a = X_sol[j,1]
        ruta['Salesman_' + str(i+1)].append(a)
        j = np.where(X_sol[:,0] == a)  
        print(j)
        j = j[0][0]
        a = j  
        print(a)  
              
                  

# Showing the paths
for i in ruta.keys():
    print('The path of ' + i + ' is:\n')
    print( ' => '.join(map(str, ruta[i])))
    print('')
    
################################################
# Plotting the optimal path
################################################

# Plotting the points
fig, ax = plt.subplots(figsize=(14,7))

for i in range(n):
    ax.annotate(str(i), xy=(xy_cords[i,0], xy_cords[i,1]+0.1))

ax.scatter(xy_cords[:,0],xy_cords[:,1])
for i in ruta.keys():
    ax.plot(xy_cords[ruta[i],0], xy_cords[ruta[i],1], label = i)
    ax.legend(loc='best')
    
# Showing the optimal distance
distance = np.sum(np.multiply(C, X.value))
print('The optimal distance is:', np.round(distance,2), 'm')
plt.show()
        