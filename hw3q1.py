# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:39:18 2022

@author: Dalanda
"""

"""
Problem 1:  Y = F(K, L) = A(K)**alpha (L)**1-alpha

Find: -  Aggregate output (Y ), given K = 10.0 and L = 50.0
      -  Marginal productivity of capital given (K, L) = (2n, 50.0), where n = 1, 2,... 10.
      -  Marginal productivity of labor given (K, L) = (10.0, 10n), where n = 1, 2,... 10.
"""
#import libraries
import numpy as np
import shutil
import pandas as pd
import numdifftools as nd

# Define & print dash so when we print we get a separtion with other questions
dash = "-" *80
print(dash)
# add a space
print()
# Get the terminal's size
columns = shutil.get_terminal_size().columns
# Start question 1 output
print("Question 1 Solutions:".center(columns))
print ()

# Assuming A= 10 and alpha=0.3
A = 10
O = 0.3
 
#define the production function in respect to K and L
def f(K, L):
    return (A * (K)**(O) * (L)**(1-O)) #return the function

#Print 1.1 answer in the center
print("The aggregate output Y given K=10 and L=50 is:".center(columns))
print()
#  changing the production output into a string so that i can also center the answer
print(str(f(10, 50)).center(columns))

# Defining the Jacobian function since we are using the central difference approximation 
def jacobian(f, x, dx, method=None): 
    # Generic form so we can use in questions 1.2 & 1.3
    n = len(x)
    jac = np.zeros([n,n])
    e = np.eye(n) * dx

# Double for loop     
    for i in range(n):
        for j in range(n):
            #Central differentiation equation
            jac[i][j] = (f(x+e[j])[i] - f(x-e[j])[i]) / (2*dx)
    return jac

# Defining the equation for 1.2
def f(X):
    
    # I could have write it more consciesly but here X is the array that represents the different K's
    x1 = X[0]; x2 = X[1]; x3 = X[2]; x4 = X[3]; x5 = X[4]; x6 = X[5]; x7 = X[6]; x8 = X[7]; x9 = X[8]; x10 = X[9]
    f = np.zeros(10) # return a new array
    
    #The array is the list of the equation's results for each K
    f[0] = A * x1**(O) * (50)**(1-O)
    f[1] = A * x2**(O) * (50)**(1-O)
    f[2] = A * x3**(O) * (50)**(1-O)
    f[3] = A * x4**(O) * (50)**(1-O)
    f[4] = A * x5**(O) * (50)**(1-O)
    f[5] = A * x6**(O) * (50)**(1-O)
    f[6] = A * x7**(O) * (50)**(1-O)
    f[7] = A * x8**(O) * (50)**(1-O)
    f[8] = A * x9**(O) * (50)**(1-O)
    f[9] = A * x10**(O) * (50)**(1-O)
    return f    

# Defining the K's 
X = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Setting the jacobian for question 1.2 as jac1
jac1 = jacobian(f, X, dx=1e-3)

# making jac1 as a column and call it jac
jac2 = jac1[jac1 !=0][:, None]
jac = np.round_(jac2, 2)
    

L = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50]) # Labor
n = np.arange(1, 11, 1) #n values

# Printing the result
print ()
print ()
print (" The marginal productivity of capital (MPK) given (K, L) = (2n, 50) is: ".center(columns))
print(dash)

# Tabulate the output
comTable = zip(n, X, jac, L)
df = pd.DataFrame(comTable, columns = ('n','Capital', 'MPK', 'Labor'))
print(df)



# Defining the equation for 1.2, the function in respect to labor
def f(L):
    f = np.zeros(10) # create a new array of the results of the function in respect to L
    
    # making each array index equals to the index's change in L
    f[0]= A * (10**O) * L[0] ** (1-O)
    f[1]= A * (10**O) * L[1] ** (1-O)
    f[2]= A * (10**O) * L[2] ** (1-O)
    f[3]= A * (10**O) * L[3] ** (1-O)
    f[4]= A * (10**O) * L[4] ** (1-O)
    f[5]= A * (10**O) * L[5] ** (1-O)
    f[6]= A * (10**O) * L[6] ** (1-O)
    f[7]= A * (10**O) * L[7] ** (1-O)
    f[8]= A * (10**O) * L[8] ** (1-O)
    f[9]= A * (10**O) * L[9] ** (1-O)
       
    return f # return the function's result

# defining the labor value in an array
L = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Setting the jacobian for question 1.2 as jac1
jac1 = jacobian(f, L, dx=1e-3)
# making jac1 as a column and call it jac
jac2 = jac1[jac1 !=0] [:, None]
jac = np.round_(jac2, 2)
# Setting capital as an array so it is easier to print
K = [10, 10, 10, 10, 10,10, 10, 10, 10, 10]   


#printing results for q1.3
print(dash) #dash
print ()
print ()
print (" The marginal productivity of labor given (K, L) = (10, 10n) is: ".center(columns)) #center
print(dash) #dash

# Tabulate the output
comTable = zip(n, K, jac, L) # get the values
# get values added and names
df = pd.DataFrame(comTable, columns = ('n', 'Capital', 'MPL', 'Labor'))
pd.set_option('display.colheader_justify', 'center') # center
print(df) #print
print(dash)  # dash to make kniwn that it is over

def jacobian(f, x, dx):
    n = len(x)
    jac = np.zeros([n,n])
    e = np.eye(n) * dx
    
    for i in range(n):
        for j in range(n):
            jac[i][j] = (f(x+e[j])[i] - f(x-e[j])[i]) / (2*dx)
    return jac


# Define & print dash so when we print we get a separtion with other questions
# add a space
print()
# get the terminal's size
columns = shutil.get_terminal_size().columns
#print(dash)
# add a space
print()
# get the terminal's size
columns = shutil.get_terminal_size().columns 
print("Question 2 Solutions".center(columns)) #question 2 centered
print () #space

#defining the function where x[0] is x anf x[1] is y
def f(x):
     f = np.zeros(2) #using numpy f is returning 2 array
     f[0] = x[0]**0.2 + x[1]**0.2 - 2   #first equation
     f[1] = x[0]**0.1 + x[1]**0.4 - 2   #second equation
     return f #return the 2 array of f

# Define the newton function with 4 parameters
def newton(f, xguess,jacobian, tol, maxiter):
 
    print()
    print("iteration         x                  y")
    print(45*"-")

    #for loop that goes until the max iteration +1
    for i in range(maxiter+1): 
              
        Z = f(xguess) 
        jac = jacobian(f, xguess,dx=1e-7)        
        x0 = np.linalg.solve(jac, -Z) #dx        
        x1 = xguess + x0

        print("%4d %18.7f %18.7f" % (i+1, x1[0], x1[1]))
             
        if np.linalg.norm(x0) < tol:
            break
        
        else:
            xguess = x1

    print(45*"-")
    return x1

#check tolerance 
#read each step 

# defening the main function that allows us to write our input into the newton's function
def main():

    x0 = [2.0, 2.0] #x,y guess
    tol = 0.000001  #tolerance input 1e-6
    maxiter = 6    #maximum of iteration

    #new value of the newton's function output 
    #calling the function putting our values
    sol = newton(f, x0, jacobian, tol, maxiter) #change one of the statement 
    
    print()
    print()
    print("The Optimal solution for is:", sol)  #print the solution

main()