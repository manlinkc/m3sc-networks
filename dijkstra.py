"""Dijkstra algorithm implemented using dictionaries.
Note: Implementation is different from lecture 8 slides but similar
to the in-class example.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def flux(G):
    tau=2
    A=nx.adjacency_matrix(G)
    A=A.todense()

    #make q into list
    #q=list(dict(nx.degree(G)).values())
    #make q into a numpy array

    q=np.fromiter(dict(nx.degree(G)).values(), dtype=np.float)
    #print(q)
    diagq=np.diag(q)
    sumterms=np.diagflat(np.reciprocal(q*A))
    F=tau*diagq*A*sumterms



    #constructing flux adjacency_matrix
    #maybe put this into one loop since num of columns=nm of rows
    #F=A.astype(float)
    #Another way of doing this is take q and make a 3x3 matrix with q in it's rows
    #for row in range(N):
    #    F[row,:]=q[row]*F[row,:]
    #Another way to do this is to use np.transpose(np.multiply(q,np.transpose(A)))

    #Use numpy.multiply for efficiency


    #F=np.multiply(sumterms,F)

    #for column in range(N):
    #    bottomsum=float(np.dot(q,A[:,column]))
    #    F[:,column]=np.divide(F[:,column],bottomsum)

    return F



def RHS(y,F,N):
    """Compute RHS of model at time t
    input: y should be a 3N x 1 array containing with
    y[:N],y[N:2*N],y[2*N:3*N] corresponding to
    S on nodes 0 to N-1, I on nodes 0 to N-1, and
    V on nodes 0 to N-1, respectively.
    output: dy: also a 3N x 1 array corresponding to dy/dt

    Discussion: add discussion here
    """
    params=(50,80,105,71,1,0.01)
    a,theta0,theta1,g,k,tau=params
    t=0.25
    theta=theta0+theta1*(1-np.sin(2*np.pi*t))

    dy=[0]*(3*N)

    dsdt= a*np.array(y[N:2*N])-(g+k)*np.array(y[0:N])+np.dot(F,y[0:N])-np.multiply(F.sum(axis=0),y[0:N])
    dy[:N]=dsdt.tolist()[0][:N]

    didt=theta*np.multiply(y[:N],y[2*N:3*N])-(k+a)*np.array(y[N:2*N])+np.dot(F,y[N:2*N])-np.multiply(F.sum(axis=0),y[N:2*N])
    dy[N:2*N]=didt.tolist()[0][:N]

    dvdt=k*(np.ones(N)-y[2*N:3*N])-theta*np.multiply(y[:N],y[2*N:3*N])+np.dot(F,y[2*N:3*N])-np.multiply(F.sum(axis=0),y[2*N:3*N])
    dy[2*N:3*N]=dvdt.tolist()[0][:N]

    print(dy)
    #dy = 0 #modify
    #return dy


def RHS1(y1,F,N):
    """Compute RHS of model at time t
    input: y should be a 3N x 1 array containing with
    y[:N],y[N:2*N],y[2*N:3*N] corresponding to
    S on nodes 0 to N-1, I on nodes 0 to N-1, and
    V on nodes 0 to N-1, respectively.
    output: dy: also a 3N x 1 array corresponding to dy/dt

    Discussion: add discussion here
    """
    params=(50,80,105,71,1,0.01)
    a,theta0,theta1,g,k,tau=params
    t=0.25
    theta=theta0+theta1*(1-np.sin(2*np.pi*t))

    dy=np.zeros((3*N,1))

    dsdt= a*y1[N:2*N]-(g+k)*y1[0:N]+F*y1[0:N]-np.multiply(np.transpose(F.sum(axis=0)),y1[0:N])
    dy[:N]=dsdt

    didt=theta*np.multiply(y1[:N],y1[2*N:3*N])-(k+a)*y1[N:2*N]+F*y1[N:2*N]-np.multiply(np.transpose(F.sum(axis=0)),y1[N:2*N])
    dy[N:2*N]=didt

    dvdt=k*(1-y1[2*N:3*N])-theta*np.multiply(y1[:N],y1[2*N:3*N])+F*y1[2*N:3*N]-np.multiply(np.transpose(F.sum(axis=0)),y1[2*N:3*N])
    dy[2*N:3*N]=dvdt

    print(dy)

if __name__=='__main__':

    #G=nx.Graph()
    #edges=[[0,1],[0,2],[1,3],[2,3],[4,5]]
    #edges=[[0,1],[0,2],[1,2],[1,3],[3,4]]
    #G.add_edges_from(edges)
    #nx.draw(G,with_labels=True)
    #plt.show()

    #Intialize a graph
    G=nx.Graph()
    #Define edges
    edges=[[0,1],[0,2],[1,3],[2,3],[4,5]]
    #Add edges
    G.add_edges_from(edges)
    #Get the adjacency matrix
    A=nx.adjacency_matrix(G)
    #Collapse it into matrix form
    A=A.todense()

    #Define Tau
    tau=0.01

    #Get the number of nodes
    nodes=G.nodes()
    N=len(nodes)

    #Get the connections of each node
    q=np.fromiter(dict(nx.degree(G)).values(), dtype=np.float)

    #Construct F
    diagq=np.diag(q)
    sumterms=np.diagflat(np.reciprocal(q*A))
    F=tau*diagq*A*sumterms

    #Vector of initial conditions
    y= [0.05, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0.1, 1, 1, 1, 1, 1]
    x=0
    #Vector y1
    y1=np.zeros(3*N)
    y1[2*N:]=1
    y1[x],y1[N+x],y1[2*N+x]=0.05,0.05,0.1
