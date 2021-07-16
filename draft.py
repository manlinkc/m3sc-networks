import numpy as np
import collections
import time

def a0min2(A,amin,J1,J2):
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
        print("moved node", nmin)

        #Update provisional distances for unexplored neighbors of nmin
        for k in A[nmin]:
            n = k[0]
            w = k[1]
            if n in Udict:
                #compute provisional distance to adjacent node, change division to multiplication by reciprocal for efficiency
                acomp = max(a0min,amin/w)
                #If provisioanl distance is smaller than the one in the unexplore dictionary then replace it
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
    A=[[(1,0.2),(2,0.8)],[(0,0.2),(3,0.3)],[(0,0.8),(3,0.1),(4,0.5)],[(1,0.3),(2,0.1)],[(2,0.5)]]
    A = [[(1,0.2),(2,0.8)],[(0,0.2),(3,0.3)],[(0,0.8),(3,0.1),(4,0.5)],[(1,0.3),(2,0.1),(5,0.2)],[(2,0.5),(5,0.1)],[(3,0.2),(4,0.1)]]
    a0=10
    amin=2
    J1=3
    J2=5
