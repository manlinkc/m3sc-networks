"""M345SC Homework 2, part 1
Manlin Chawla CID:01205586
"""
import numpy as np
import collections

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

    Shortest number of days needed to assemble the phone:
    The process consists of N tasks and M of these N tasks require at least one other
    task to have been completed before the task itself can be started. This means
    there are N-M tasks that can be completed on day 0. All the tasks can be represented
    as nodes in a network. For simplicity I will refer to the N-M tasks completed on
    day 0 as the initial nodes. The remaining M tasks depend on at least one other
    task to be completed, the dependencies can be represented in the network as path
    connecting the initial node to dependent nodes to the current node itself. For
    example, if task 15 depends on tasks 1,4,9 and 12 being completed then there is a
    path connecting node 1 to node 4 to node 9 to node 12 to node 15. Overall the
    shortest number of days needed to assemble the phone is the same as the maximum
    of the lengths of all the paths, which is the maximum depth of the network.

    Implementation of scheduler(L):
    The basic approach for the scheduler function is to iterate through L and construct
    a list (Lcounter) which contains the lengths of each sub-list in L. An empty sublist
    for a task means that this task has no dependencies. All of these tasks can be scheduled
    to be completed on day 0 and are marked as explored (in the Lexplored list). All day 0
    tasks are pooled together and added to a que. For each element in the que, the algorithm
    iterates through L to find the tasks that depend on this element. If a task depends on
    this element in the que then the counter is reduced by one. Now check the updated counterlist,
    any task that now has 0 length can be scheduled to be completed on day 1 and S is updated
    with this information. The day 1 tasks form the new updated que. This process is repeated
    again to find tasks which are to be scheduled to be completed on day 2, day 3, day 4
    onwards, until all tasks have been explore and assigned to a day.

    Time Complexity
    The analysis of the running time can be divided into three parts – 1) the cost of setting up
    the lists, que and counters to be used in the algorithm, 2)  the cost of finding the lengths
    of all the sublists in L, and finding the tasks that should be assigned to day zero and
    3) the cost of finding what days the tasks should be assigned on day 1 onwards.

    1)	The details for the asymptotic running time for the setup portion of the code are as below:
        O(1) : Calculating the number tasks
        O(N) : Initializing a list of zeros to track which tasks/nodes have been explored (Lexplored)
        O(N) : Initializing a list of zeros to contain the day that each task is completed on (S)
        O(N) : Initializing a list of zeros to contain the lengths of each sub-list in L (Lcounter)
        O(1) : Initializing an empty list for the que (dependencies)
        O(1) : Initializing a counter that records how many tasks are left to explored (numoftasksleft)
        O(1) : Initializing a counter for the current day

        Overall the leading order time for the setup is O(N)

    2) 	The details for the asymptotic running time for finding the day 0 tasks and lengths of
        each sub-list portion of the code are as below:
        O(N) : A loop iterates through the N sub-lists in L to find which tasks should be assigned to day 0.
        Within this loop:
            O(1):  Checking if the sub-list is empty
            O(1):  If the sub-list is empty then the tasks is appended to the end of the que,
            the explored status is updated and the counter for the number of tasks left to explore is reduced by 1
            O(1):   If the sub-list is not empty then the length of the sub-list is computed, the cost of the len() function is O(1)

	     Overall the leading order time for the setup is O(N)

    3) In the next part of the algorithm enters a while loop where it first checks if the
        counter for the number of tasks left to see if there are any tasks left to explore,
        this step has O(1) complexity. In the best case M=0 and all N tasks can be completed
        on day 0 and this would be the end of the algorithm. In the worst-case scenario,
        this algorithm remains in the while loop for N-1 iterations so this loop has a time
        complexity of O(N-1) in the worst case.

        Within this while loop there is another nested while loop for iterating through
        the que, all tasks that were assigned to be completed on the previous day.  For
        the code to even reach this point is clear that M>=1. In the worst case there will
        be N-1 elements in the que and so the algorithm will remain in this nested while loop
        for N-1 iterations implying that the worst-case time complexity is O(N-1). Within this
        nested while loop first we isolate/extract the first element for the Q,  as explained
        above in the worst case the Q would have N-1 elments so popping from the front would
        be an O(N-1), I am using the collections module and using the popleft() function which
        reduces the time complexity for this step down to O(1).

        Next there is a nested for loop which iterates through all sublist in L to see if
        this task depends on an element in Q. This loop has O(N) time complexity. The task is
        checked to see whether it is explored or not, this step has O(1) complexity. In the
        worst case the if statement will be satisfied (N-1)/2 so the next loop repeats (N-1)/2 times.
        There is a nested for loop which searches through the sublist to see if the completed task
        from the Q appears in the sub-list. On average searching through a list this step has a
        complexity of O(N/2) complexity. If the element is found the counter is updated which is
        O(1) and the nested loop breaks as there is no need to check the rest of the elements.

        Next there is a loop which iterates through the updated Lcounter and Lexplored lists.
        This step has O(N) complexity. To update S we only need to check unexplored tasks that
        have counter that is now 0, checking these conditions are achieved is O(1) time complexity.
        Updating the Lexplored list, S and the number of tasks left counter are all O(1).

    Overall worst-case complexity arises from a scenario where task N depends on task N-1 which
    depend on task N-2 …. which depends on task 1 which depends on task 0. In this case ,the
    worst case complexity comes from the following contributions:

    O(N) for the setup
    O(N) for the finding day 0 taks and setting up counter list
    O(N-1) for the outer while loop
    O((N-1)/2) for the nested for loop that searches through only the sublists which are unexplored
    O(N/2) for the searching
    O(N) updating S, Lexplored and Q before the next iteration

    Therefore, the leading term for the worst case asymptotic time is O(N^3).

    Efficiency
    There are several ways where I have tried to make my implementation as efficient as possible.
    Another possible implementation of scheduler could be to find all tasks completed on day 0,
    pool these in a Q and then iterate through L removing the day 0 tasks from the sublist,
    using pop() and then computing the new length of each sublist and assigning it a day.
    I found this method to be inefficient for some cases as it would require popping elements
    from the middle of a list, the collections module doesn’t have a function that facilitates
    popping from the middle of a list so this step would have worst case complexity O(K) where K
    is the length of the sublist. To avoid this problem, I used counter lists which represent the
    length of a list as it progresses through the algorithm without having to pop anything.

    I have also made my code efficient by having a counter to represent the number of tasks left.
    I could use a statement which iterates through Lexplored to see if any tasks are unexplored at
    the start of each loop but this would have O(N) complexity. Instead the counter reduces this step
    to O(1) complexity. I have used the collections module to deque and pop from the beginning of the
    list with O(1) complexity and used breaks and if statements to carry out steps only when necessary.

    """
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
    # Initialize a counter that records how many tasks are left to explore
    numoftasksleft=N

    #------------------------------DAY 0------------------------------------------------

    # Iterate through L to find tasks that can be assigned to day 0
    for i,sublist in enumerate(L):
        # If the sublist is empty it has no dependencies and can be completed on day 0
        # Check if sublist is empty
        if not sublist:
            # Append task to the depencies tasks que
            dependencies.append(i)
            # Mark task as explored
            Lexp[i]=1
            # Lcounter and S already contain 0's no need to explicitly update length and day
            # Lcounter[i]=0
            # Update the number of tasks left
            numoftasksleft += -1

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
    while numoftasksleft>0:
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
        #--------------------------------UPDATE S-------------------------------------

        # If the Lcounter for a task is now 0 then this task can be assigned to be completed on the current day
        # Update S
        for m in range(N):
            # Only need to check unexplored tasks that have counter that is now 0
            if Lexp[m]==0 and Lcounter[m]==0:
                    # Mark task as explored
                    Lexp[m]=1
                    # Update S
                    S[m]=day
                    # Update dependencies Q for the next interation of the loop
                    dependencies.append(m)
                    # Update the number of tasks left to explore
                    numoftasksleft += -1

        # Update the day counter for the next iteration of the loop
        day+=1

    return S

def findPath(A,a0,amin,J1,J2):
    """
    Question 1.2 i)
    Search for feasible path for successful propagation of signal
    from node J1 to J2

    Input:
    A: Adjacency list for graph. A[i] is a sub-list containing two-element tuples (the
    sub-list my also be empty) of the form (j,Lij). The integer, j, indicates that there is a link
    between nodes i and j and Lij is the loss parameter for the link.

    a0: Initial amplitude of signal at node J1

    amin: If a>=amin when the signal reaches a junction, it is boosted to a0.
    Otherwise, the signal is discarded and has not successfully
    reached the junction.

    J1: Signal starts at node J1 with amplitude, a0
    J2: Function should determine if the signal can successfully reach node J2 from node J1

    Output:
    L: A list of integers corresponding to a feasible path from J1 to J2.

    Discussion: Add analysis here

    Implementation:
    The basic approach I used for the findPath function is a modified version of breadth
    first search that uses J1 as the source node. The algorithm maintains a list of nodes,
    a list of explored/unexplored labels for the nodes and a queue. It first initializes
    the queue with the source node and marks it as explored. Then it removes the nodes from
    the queue in the order they were added (first in first out). Then search through the edges
    of the removed nodes. Here the unexplored nodes are only added to the queue if the amplitude
    a0 multiplied by the loss Lij is greater than the threshold amin. If a node is added to
    the queue it is marked as explored. The search is terminated early if the target node J2
    is found (meaning a viable path is found) or if the queue is empty (meaning a path has
    not been found). In addition to retrieving the path, I have used another list which at
    the index corresponding to a node, stores the previous node in the path. We can use this
    list to back track through the path the algorithm took to reach from J2 to J1 and then
    reverse it to get the final output path J1 to J2.

    Efficiency
    Deciding whether a BFS or DFS implementation is better heavily depends on the structure
    of the graph and the location of the J2 node. BFS is better/faster if the J2 node is not
    far from J1 in the network. As J2 will only appear once in the network, if the tree is
    really deep the DFS might take an extremely long time but the BFS could be faster. In
    this problem we are only looking for one feasible path, given that there might be some
    router where the threshold might not be met, BFS can quickly find one which is viable.
    DFS would be better if we wanted to exhaust all the possibilities/find all paths.

    To ensure that the modified BFS algorithm I have used for the findPath function is as efficient
    as it can be, I have used the collections module to use the popleft function to remove elements
    from front of the que with O(1) operations. This is better than using the inbuilt python pop
    function which requires shifting all other elements in the que and has O(N) complexity.

    When it comes to getting the path, storing a path to each node from a source node in a
    list doesn’t effect the cost estimate. As the findPath algorithm iterates through the graph
    layer by layer we can maintain a list of paths. However, this can be memory inefficient for
    large graphs/networks as we are only really interested in a path from J1 to J2. Instead, I
    have a list which stores the previous node searched for each explored node which if more
    memory efficient. However, this requires an extra loop to back track the path.

    Running Time
    In general, if we are using an adjacency list that represents a network with N nodes and M
    edges then the worst case cost of a general BFS algorithm is O(M+N). To get the time complexity
    of my implementation of findPath I am considering the extra steps I have added that modify a
    generic BFS algorithm. For the findPath algorithm if the signal amplitude falls below the
    threshold then the signal is removed from the network and doesn’t reach the junction. Evaluating
    the condition a0xLij >=amin is an O(1) operation. Appending to que is an O(1) operation as I
    am using the collections module. There will be some nodes which BFS does not need to explore.
    In the worst case we can have a network where the signal never fail and passes along all the
    edges between two nodes. In addition to this we have the worst case where BFS searches through
    the entire network and J2 is the last node remaining in the network to explore. In the worst
    case the algorithm has a complexity of O(M+N) up until this point in the code. Next for back
    tracking to finding the path has worst case complexity O(N). I got to this estimate by considering
    a worst case scenario where the graph has a straight line/chain form e.g. for example node 0
    is connected to node 1 which is connected to node 2 which is connected to node 3 etc. Back tracking
    for the path will be O(N) operations and another O(N) operations for reversing the list to form the
    output. Overall the leading order term for the asymptotic running time of this implementation in
    the worst case is O(M+N).

    """


    numnodes=len(A)

    L1 = list(range(numnodes)) #Assumes nodes are numbered from 0 to N-1
    L2 = [0]*numnodes #explored or unexplored
    L4 = [0]*numnodes
    L=[]

    Q = collections.deque([J1])
    L2[J1] = 1
    L4[J1] = [J1]
    pathfound = 0
    while len(Q)>0:
        x = Q.popleft()
        if x==J2:
            pathfound = 1
            break

        for v in A[x]:
            if L2[v[0]]==0 and a0*v[1]>=amin:
                    L2[v[0]]=1
                    Q.append(v[0])
                    L4[v[0]]=x

    if pathfound==1:
        L=[J2]
        while x!=J1:
            x=L4[x]
            L.append(x)
        L.reverse()

    return L

def a0min(A,amin,J1,J2):
    """
    Question 1.2 ii)
    Find minimum initial amplitude needed for signal to be able to
    successfully propagate from node J1 to J2 in network (defined by adjacency list, A)

    Input:
    A: Adjacency list for graph. A[i] is a sub-list containing two-element tuples (the
    sub-list my also be empty) of the form (j,Lij). The integer, j, indicates that there is a link
    between nodes i and j and Lij is the loss parameter for the link.

    amin: Threshold for signal boost
    If a>=amin when the signal reaches a junction, it is boosted to a0.
    Otherwise, the signal is discarded and has not successfully
    reached the junction.

    J1: Signal starts at node J1 with amplitude, a0
    J2: Function should determine min(a0) needed so the signal can successfully
    reach node J2 from node J1

    Output:
    (a0min,L) a two element tuple containing:
    a0min: minimum initial amplitude needed for signal to successfully reach J2 from J1
    L: A list of integers corresponding to a feasible path from J1 to J2 with
    a0=a0min
    If no feasible path exists for any a0, return output as shown below.

    Discussion: Add analysis here

    Implementation
    The basic approach I used for the a0min algorithm is to find a provisional a0
    that will work along a path from J1 to J2 (max of the (a0 along all edges in that path)).
    And then take the minimum of all of the provisional a0’s to get the final value to output.
    To do this I have used a modified Dijkstra’s algorithm. A sketch of the algorithm is as follows:
    First two dictionaries are setup to store the explored and unexplored nodes and the provisional values
    of a0’s are set to be an arbitrary large number. Then the source node is labelled as
    explored and  the code finds the provisional a0 values for all neighbours of the
    source by computing a0=amin/Lij. Updating/replacing the value in the unexplored
    dictionary is done if and only if the new provisional a0 is smaller than the one
    already being stored in the unexplored dictionary. In each iteration we find the
    unexplored node that has the smallest provisional a0. Label this as explored and
    move it to the explored dictionary and update the provisional distance of all the
    neighbours. The algorithm continues exploring until no reachable nodes remain unexplored
    or until J2 is reached. If J2 is reached then we back track to return the path and
    the minimum a0.

    Running Time:
    For a general Dijkstra’s algorithm for a graph the worst case running time is O(N^2)
    operations. For my implementation of a0min the extra steps I have added to modify the
    Dijkstra’s algorithm have the following time complexities. Firstly setting up the two
    dictionaries is O(1) operations. Initializing with an arbitrary large value is O(N).
    Within each iteration the a0 is calculated for each neighbour and the maximum of the
    provisional a0’s is taken. These steps has O(N) operations overall.

    Once J2 has been reached in the graph the path is retrieved. During the algorithm I
    have used another list which at the index corresponding to a node, stores the previous
    node in the path. We can use this list to back track through the path the algorithm took
    to reach from J2 to J1 and then reverse it to get the final output path J1 to J2 with the
    best/most desirable a0. This is more memory efficient than storing the paths from J1 to all
    the nodes in the network in a list containing lists.

    In the worst case back tracking the path is O(N) operations and reversing the list is O(N)
    operations. Once the path is found the best a0, path is returned code breaks as there is no
    reason for the algorithm to continue. Overall the leading order term for my implementation of
    a0min function is O(N^2).


    Efficiency
    In the a0min implementation for each iteration there are O(N) operations to find the
    provisional a0 values. Depending on the type of graph, a binary heap is a more efficient
    and provides these operation in O(log base 2 N) time. A binary heap arranges the nodes
    in a list and the order of the nodes is determined by the provisional a0 values. The
    arrangement of the elements in the list correspond to a binary tree. To use a binary
    heap this would require using the Python heapq module rather than dictionaries or lists.
    When the smallest provisional a0 value is popped from the heap, the list he restructured
    in O(N log base 2 N) time. Then if needed we’d have to adjust the provisional a0 of the
    neighbours of the removed provisional a0 node and restructure the heap which has
    O(M log base 2 N) complexity.

    An a0min + binary heap algorithm would have O(N*Mlog(N)) worst case time complexity.
    This is generally better than my implementation of the a0min algorithm which has O(N^2)
    complexity. However for graphs where M is close to N this worse case complexity become
    approximately O(N^2log(N)) and in this case it would be worse to use a binary heap
    implementation for the a0min function.

    """
    output = -1,[]

    #Initialize dictionaries
    ainit = 10**6
    Edict = {} #Explored nodes
    Udict = {} #Unexplored nodes

    numnodes=len(A)
    #path = [[] for L in range(numnodes)]
    path=[0]*numnodes

    for n in range(numnodes):
        Udict[n] = ainit
    Udict[J1]=0
    #path[J1].append(J1)
    #Main search
    while len(Udict)>0:
        #Find node with min d in Udict and move to Edict
        a0min = ainit
        #for key value in the dictionary
        for n,a in Udict.items():
            #in each iteration only one node will have a value less than 10**6 so dmin is replaced once
            #d min replaced, if a smaller one comes along then it is replaced until you only pick the smallest possible
            if a<a0min:
                a0min=a
                nmin=n

        #Move the node with minimum a0 to the explored dictionary
        Edict[nmin] = Udict.pop(nmin)
        #print("moved node", nmin)

        #Update provisional a0 for unexplored neighbors of nmin
        for k in A[nmin]:
            n = k[0]
            w = k[1]
            if n in Udict:
                #compute provisional a0 to adjacent node, change division to multiplication by reciprocal for efficiency
                acomp = max(a0min,amin/w)
                #If provisioanal a0 is smaller than the one in the unexplore dictionary then replace it
                if acomp<Udict[n]:
                    Udict[n]=acomp
                    path[n]=nmin
                    #if path[n]:
                        #if path[n][-1]!=nmin:
                        #path[n].clear()
                    #path[n].extend(path[nmin])
                    #path[n].append(n)
        if nmin==J2:
            L=[J2]
            x=J2
            while x!=J1:
                x=path[x]
                L.append(x)
            L.reverse()
            output=Edict[J2],L
            break

    return output

if __name__=='__main__':
    L=None
