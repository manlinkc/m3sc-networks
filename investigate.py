def modelNinvestigate(G,x=0,params=(50,80,105,71,1,0.01),tf=6,Nt=400):
    """
    Question 2.1
    Simulate model with tau=finite positive

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    tf,Nt: Solutions Nt time steps from t=0 to t=tf (see code below)
    display: A plot of S(t) for the infected node is generated when true

    x: node which is initially infected

    Output:
    Smean,Svar: Array containing mean and variance of S across network nodes at
                each time step.
    """
    #Should I reduce the RHS formula's by evaluating the zeros?No just keep the how they are

    #Use matmul to make everything efficient
    a,theta0,theta1,g,k,tau=params
    tarray = np.linspace(0,tf,Nt+1)
    Smean = np.zeros(Nt+1)
    Svar = np.zeros(Nt+1)

    nodes=G.nodes()
    N=len(nodes)
    #A=nx.adjacency_matrix(G)
    A = nx.to_scipy_sparse_matrix(G)
    #Collapse it into matrix form
    A=A.todense()

    #Get the connections of each node
    #q=np.fromiter(dict(nx.degree(G)).values(), dtype=np.float)
    q=A.sum(axis=0,dtype=float).tolist()[0]
    #print(q)

    #Construct F
    #diagq=np.diag(q)
    diagq=scipy.sparse.diags(q)
    diagq=diagq.todense()
    #print(diag(q))
    #sumterms=np.diagflat(np.reciprocal(q*A))
    sumterms=scipy.sparse.diags(np.reciprocal(q*A))
    sumterms=sumterms.todense()
    F=tau*diagq*A*sumterms
    print(np.sum(np.sum(F)))

    #Set up vector y to input in RHS
    y=np.zeros(3*N)
    y[2*N:]=1
    y[x],y[N+x],y[2*N+x]=0.05,0.05,0.1
    print('y=',y)
    #Add code here
    dy=np.zeros(3*N)
    Fsum=np.sum(F,axis=0)

    print(np.shape(F))

    def RHS2(y,t):
        """Compute RHS of model at time t
        input: y should be a 3N x 1 array containing with
        y[:N],y[N:2*N],y[2*N:3*N] corresponding to
        S on nodes 0 to N-1, I on nodes 0 to N-1, and
        V on nodes 0 to N-1, respectively.
        output: dy: also a 3N x 1 array corresponding to dy/dt

        Discussion: add discussion here
        """
        theta=theta0+theta1*(1-np.sin(2*np.pi*t))

        dsdt= a*y[N:2*N]-(g+k)*y[0:N]+np.matmul(F,y[:N])-tau*y[:N]
        dy[:N]=dsdt

        didt=theta*np.multiply(y[:N],y[2*N:3*N])-(k+a)*y[N:2*N]+np.matmul(F,y[N:2*N])-tau*y[N:2*N]
        dy[N:2*N]=didt

        dvdt=k*(1-y[2*N:3*N])-theta*np.multiply(y[:N],y[2*N:3*N])+np.matmul(F,y[2*N:3*N])-tau*y[2*N:3*N]
        dy[2*N:3*N]=dvdt

        #dy = 0 #modify
        return dy

    y=odeint(RHS2,y,tarray)


    #return
    return N,y,tarray
