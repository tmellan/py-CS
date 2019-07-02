##################
#Support functions
##################
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

#function soft(x,l)
#    sign(x).*plusOp(abs(x)-l)
#end
