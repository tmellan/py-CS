#############################################
##############Support Functions##############
#############################################

import numpy as np
from numpy.linalg import inv

def findSol(Y,A,M,Mt):
    return A-np.matmul(Mt,np.matmul(M,A)-Y)

def initSolution(MeasurementMatrix,MeasuredOutput,m):
    #Guess a random solution
    GuessedInput=np.zeros(m)
    GuessedInput[(np.random.uniform(0,1,round(m/10))*(m-1)+np.ones(round(m/10))).astype(int)] = np.random.uniform(0,1,round(m/10))

    #project the guessed solution into the space of exact answers
    GuessedInput=findSol(MeasuredOutput,
        GuessedInput,
        MeasurementMatrix,
        np.linalg.pinv(MeasurementMatrix))
    return GuessedInput

def plusOp(x):
    x.clip(min=0)

#############################################
#Iteratively re-weighted least squares (IRLS)
#############################################

def IRLS(MeasurementMatrix,MeasuredOutput,verbose="false",maxiter=1000,p=.5,threshold=1e-7,debug="false"):

#Smooth
    eps=lambda eps : 1/eps**3
    
# identify the size of the input
    m=MeasurementMatrix.shape[1]
    
# First we need to calculate a valid solution
    GuessedInput = initSolution(MeasurementMatrix,MeasuredOutput,m)
    
###### Initialize some values used in the IRLS algorithm ############
#Construct the weight matrix, used in the ridge regression
    wn = np.zeros([m,m])
    
#Save a copy of the "previous" guess, which in this case is the original guess
    PrevGuess=GuessedInput

#transpose the sampling matrix, this is used every iteration so it
#is far more efficient to calculate it before hand.  
    tMeasurementMatrix=MeasurementMatrix.transpose()
    
#set the distance between iterations to infinite    
    PrevDist = np.full(round(maxiter/100 + 1), np.inf)
    
#start on iteration 1
    iteration=1
    if verbose == "true":
        print("Iteration: \n", iteration)
          
#assume we are converging, this will be set to false if neccessary
    converges="true"
    
    debugCounter = 0
    if debug == "true":
        print("\n 1."+str(debugCounter)+" Starting while loop")

    while PrevDist[round(iteration/100)] > threshold:
        
        debugCounter +=1

        if debug == "true":
            print("\n 1."+str(debugCounter)+" Starting for loop for wn")
            
        for j in range(0,m):
            wn[j,j] = 1. / (GuessedInput[j]**2  + eps(iteration))**(p/2. - 1.)
            
        if debug == "true":
            print("\n 2."+str(debugCounter)+" Ended for loop for wn")
        
        PrevGuess=GuessedInput
        
        if debug == "true":
            print("\n 3."+str(debugCounter)+" PrevGuess set as GuessInput\n Starting IRLS step")
    
#        #IRLS step
#        GuessedInput=wn*tMeasurementMatrix*
#        pinv(MeasurementMatrix*wn*tMeasurementMatrix)*
#        MeasuredOutput
            
        wntMeasurementMatrix = np.matmul(wn,tMeasurementMatrix)  
        MMwntMMpinv = np.linalg.pinv(np.matmul(MeasurementMatrix, wntMeasurementMatrix))  
        GuessedInput = np.matmul(wntMeasurementMatrix,np.matmul(MMwntMMpinv,MeasuredOutput))
        
        if debug == "true":
            print("\n 4."+str(debugCounter)+" Ending IRLS step")
       
        if iteration % (maxiter/100) == 1:
        
        #Measure Euclidean distance between answers  
            PrevDist[round(iteration/100)+1] = np.sqrt(np.sum(np.square(PrevGuess-GuessedInput)))/m
        
            if verbose == "true":
                print("\n"+str(iteration)+ " Euclidean Distance between steps: "+str(PrevDist[int(round(iteration/100)+1)]) )
        
        #most involved convergence test, see if the distance increases from one iteration
        #to the next, and if it does, break, converges=false since we didn't hit threshold
            if ((round(iteration/100)+1>2) and (PrevDist[round(iteration/100)+1] > PrevDist[max(round(iteration/100)+1,1)])):
                converges="false"
                break
            
        #if we pass maxiter iterations, give up
            if iteration>=maxiter:
                converges="false"
                break
            
        iteration+=1
            
    if verbose == "true":
        print("Completed Iteration: \n", iteration)
            
    #if debugging, return additional information, such as distance to convergence and iteration number
    if debug == "true":
        return (GuessedInput,converges,iteration-1,PrevDist[0:int(iteration/100)+1])
#       print(GuessedInput,converges,iteration-1,PrevDist[0:int(iteration/100)+1])
    else:
        return GuessedInput
#       print(GuessedInput)
