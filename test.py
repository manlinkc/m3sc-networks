import time
import numpy as np
import collections

def scheduler3(L):
    """
    Question 1.1
    Schedule tasks using dependency list provided as input

    Input:
    L: Dependency list for tasks. L contains N sub-lists, and L[i] is a sub-list
    containing integers (the sub-list my also be empty). An integer, j, in this
    sub-list indicates that task j must be completed before task i can be started.

    Output:
    L: A list of integers corresponding to the schedule of tasks. L[i] indicates
    the day on which task i should be carried out. Days are numbered starting
    from 0.

    Discussion: Add analysis here
    """
    # Timing the code
    t1=time.time()
    #--------------------------------SETUP--------------------------------------------

    #Calculate the number of tasks
    N=len(L)
    # Intialize a list to track which tasks/nodes have been explored
    Lexp=[0]*N
    # Initalize a list to contain the number of dependencies for each task
    Lcounter=[0]*N
    # Initialize a list to contain the day that each task is completed on, used for the output
    S=[0]*N
    # Intialize a dependencies que, use the code to add the tasks that will be assigned a day in the previous iteration
    dependencies=[]

    #------------------------------DAY 0------------------------------------------------

    # Iterate through L to find tasks that can be assigned to day 0
    for i,sublist in enumerate(L):
        # If the sublist is empty it has no dependencies and can be completed on day 0
        if not sublist:
            # Append task to the depencies tasks que
            dependencies.append(i)
            # Mark task as explored
            Lexp[i]=1
            # Lcounter and S already contain 0's no need to explicitly update length and day
            # Lcounter[i]=0
        # If the sublist is not empty
        else:
            # Update Lcounter with the number of dependencies for each task
            Lcounter[i]=len(sublist)

    #----------------------------DAY 1 ONWARDS ALGORITHM-------------------------------

    # Use collections module to deque dependencies que, allows efficient popping
    dependencies = collections.deque(dependencies)

    # Counter for the days
    day=1

    # Check whether there are any tasks/nodes that are unexplored, if there are enter the loop
    while all(i == 1 for i in Lexp)==False:
        # Iterate through all tasks that were assigned to be completed on the previous day
        while len(dependencies)>0:
            # Pop from the list
            j=dependencies.popleft()
            # Iterate through all elements in L to see if the tasks from the Q appear
            for k,sublist in enumerate(L):
                # Check if task is unexplored
                if Lexp[k]==0:
                    for v in sublist:
                        # If any of the tasks in the sublist match the dependent tasks
                        if v==j:
                            # If found subtract 1 from the counter
                            Lcounter[k]=Lcounter[k]-1
                            # No need to check the rest of the list
                            break
        #--------------------------------UPDATE S FOR---------------------------------

        # If the Lcounter for a task is now 0 then this task can be assigned to be completed on the current day
        # Update S
        for m in range(N):
            # Only need to check unexplored tasks
            if Lexp[m]==0:
                # Check if the counter for the task is now 0
                if Lcounter[m]==0:
                    # Mark task as explored
                    Lexp[m]=1
                    # Update S
                    S[m]=day
                    # Update dependencies Q for the next interation of the loop
                    dependencies.append(m)

        # Update the day counter for the next iteration of the loop
        day+=1

    t2=time.time()
    print('Manlin Final Method =',t2-t1)

    return S

def scheduler2(L):
    """
    Question 1.1
    Schedule tasks using dependency list provided as input

    Input:
    L: Dependency list for tasks. L contains N sub-lists, and L[i] is a sub-list
    containing integers (the sub-list my also be empty). An integer, j, in this
    sub-list indicates that task j must be completed before task i can be started.

    Output:
    L: A list of integers corresponding to the schedule of tasks. L[i] indicates
    the day on which task i should be carried out. Days are numbered starting
    from 0.

    Discussion: Add analysis here
    """

    t1=time.time()
    N=len(L)
    Lexp=[0]*N
    S=[0]*N

    Q=[]

    #Find elements for day 0 and append to Q, reverse all other lists.
    for i,sublist in enumerate(L):
        if not sublist:
            Q.append(i)
            Lexp[i]=1

    #print('Q=',Q)
    #print('New L=',L)

    day=1
    #count=1
    while all(i == 1 for i in Lexp)==False:
        #print('wahayy=',count)
        while len(Q)>0:
            Q = collections.deque(Q)
            j=Q.popleft()
            #print(j)

            for k,sublist in enumerate(L):
                #print(k)
                if Lexp[k]==0:
                    if not sublist:
                        continue
                    else:
                        for i in range(len(sublist)):
                            if sublist[i]==j:
                                sublist.pop(i)
                                L[k]=sublist
                                break
                        #print(sublist)
            #print('Q=',Q)

        for m,sublist in enumerate(L):
            if Lexp[m]==0:
                if not sublist:
                    Lexp[m]=1
                    S[m]=day
                    Q.append(m)
        day+=1
        #count+=1

    t2=time.time()
    print('Manlin Reverse =',t2-t1)

    return S

def scheduler(L):
    """
    Question 1.1
    Schedule tasks using dependency list provided as input

    Input:
    L: Dependency list for tasks. L contains N sub-lists, and L[i] is a sub-list
    containing integers (the sub-list my also be empty). An integer, j, in this
    sub-list indicates that task j must be completed before task i can be started.

    Output:
    L: A list of integers corresponding to the schedule of tasks. L[i] indicates
    the day on which task i should be carried out. Days are numbered starting
    from 0.

    Discussion: Add analysis here
    """

    t1=time.time()
    N=len(L)
    Lexp=[0]*N
    S=[0]*N
    initmatrix=np.zeros((N,N))

    #fill matrix with dependencies
    for i,sublist in enumerate(L):
        for j in sublist:
            initmatrix[i,j]=1
    day=0
    while all(i == 1 for i in Lexp)==False:
        itermatrix=initmatrix.copy()
        #go down rows and find all empty rows
        for k in range(N):
            if Lexp[k]==0:
                if all(elt == 0 for elt in itermatrix[k,:]):
                    Lexp[k]=1
                    S[k]=day
                    #delete column in iteration matrix
                    initmatrix[:,k]=0
        day+=1
    t2=time.time()
    print('Manlin Matrix =',t2-t1)

    return S

if __name__=='__main__':
    #L=[[],[0],[0,3,13],[0],[0,1],[0,13],[0,1,4],[],[7],[7,8],[7],[0,7],[0,7,11],[0]]
    #L=[[],[],[0],[0,3,13],[0],[1]]
    L=[[],[0],[0,3,13],[0],[0,1],[0,13],[0,1,4],[],[7],[7,8],[7],[0,7],[0,7,11],[0],[0,7,11,12],[0,2,3,5,7,11,12,13,14,19,18,22],[0,1,4,6,25],[0,1,4,6,10,16,25],[0,2,3,5,13,22],[0,2,3,5,13,18,22],[],[0,2,3,13,22],[0,2,3,13],[20],[0,2,3,13,21,22],[0,1,4,6],[20,23],[20,23,26],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27,28],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27,28,29],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27,28,29,30],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27,28,29,30,31],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27,28,29,30,31,32],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27,28,29,30,31,32,33],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27,28,29,30,31,32,33,34],[0,2,3,5,7,11,12,13,14,15,19,18,22,20,23,26,27,28,29,30,31,32,33,34,35]]
    ansatz=[0,1,2,1,2,2,3,0,1,2,1,1,2,1,3,6,5,6,4,5,0,4,3,1,5,4,2,3,7,8,9,10,11,12,13,14,15]
