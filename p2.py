"""M345SC Homework 2, part 2
Manlin Chawla CID:01205586
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def eqns(d,t,a,theta0,theta1,g,k,tau):

    theta=theta0+theta1*(1-np.sin(2*np.pi*t))

    dsdt=a*d[1]-(g+k)*d[2]
    didt=theta*d[2]*d[0]-(k+a)*d[1]
    dvdt=k*(1-d[0])-theta*d[2]*d[0]

    finaldt=(dvdt,didt,dsdt)

    return finaldt


#def model1(G,x=0,params=(50,80,105,71,1,0),tf=6,Nt=400,display=False):
def model1(params=(50,80,105,71,1,0),tf=6,Nt=400,display=True):
    """
    Question 2.1
    Simulate model with tau=0

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    tf,Nt: Solutions Nt time steps from t=0 to t=tf (see code below)
    display: A plot of S(t) for the infected node is generated when true

    x: node which is initially infected

    Output:
    S: Array containing S(t) for infected node
    """

    a,theta0,theta1,g,k,tau=params
    tarray = np.linspace(0,tf,Nt+1)
    S = np.zeros(Nt+1)

    #Initial Condition
    VIS=(0.1,0.05,0.05)
    Vul,Inf,Spr=VIS

    d = odeint(eqns,(Vul,Inf,Spr),tarray,args=(a,theta0,theta1,g,k,tau,))

    S=d[:,2]

    if display==True:
        plt.figure()
        plt.plot(tarray,S)
        plt.xlabel('t')
        plt.ylabel('S')
        plt.title('Plot of S against time')
        plt.show()

    return S

def modelN(G,x=0,params=(50,80,105,71,1,0.01),tf=6,Nt=400,display=True):
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
    a,theta0,theta1,g,k,tau=params
    tarray = np.linspace(0,tf,Nt+1)
    Smean = np.zeros(Nt+1)
    Svar = np.zeros(Nt+1)

    nodes=G.nodes()
    N=len(nodes)
    A=nx.adjacency_matrix(G)
    A=A.todense()


    q=A.sum(axis=0,dtype=float).tolist()[0]

    F=np.transpose(np.multiply(q,A))
    F=tau*np.divide(F,F.sum(axis=0))
    print(np.sum(np.sum(F)))

    #Set up vector y to input in RHS
    y=np.zeros(3*N)
    y[2*N:]=1
    y[x],y[N+x],y[2*N+x]=0.05,0.05,0.1

    #Add code here
    dy=np.zeros(3*N)
    Fsum=np.sum(F,axis=0)


    def RHS(y,t):
        """Compute RHS of model at time t
        input: y should be a 3N x 1 array containing with
        y[:N],y[N:2*N],y[2*N:3*N] corresponding to
        S on nodes 0 to N-1, I on nodes 0 to N-1, and
        V on nodes 0 to N-1, respectively.
        output: dy: also a 3N x 1 array corresponding to dy/dt

        Discussion: add discussion here

        Calculating g+k is 1 operation, calculating sum2=tau*y is N operations,
        calculating a*y[N:2*N] is N operations, calculating (g+k)*y[0:N] is N operations,
        the matrix multiplication is N^2 operations, summing the four seprate terms of dsdt is
        3N operations and finally dsdt is calculated.

        The number of estimates required for calculating dsdt in one call of RHS is
        N^2+6N+1
        """
        theta=theta0+theta1*(1-np.sin(2*np.pi*t))
        sum2=tau*y
        thetaSV=theta*np.multiply(y[:N],y[2*N:3*N])

        dsdt= a*y[N:2*N]-(g+k)*y[0:N]+np.matmul(F,y[:N])-sum2[:N]
        dy[:N]=dsdt

        didt=thetaSV-(k+a)*y[N:2*N]+np.matmul(F,y[N:2*N])-sum2[N:2*N]
        dy[N:2*N]=didt

        dvdt=k*(1-y[2*N:3*N])-thetaSV+np.matmul(F,y[2*N:3*N])-sum2[2*N:3*N]
        dy[2*N:3*N]=dvdt

        return dy

    y=odeint(RHS,y,tarray)

    Smean=np.mean(y[:,:N], axis=1)
    Svar=np.var(y[:,:N], axis=1)

    if display==True:
        plt.figure()
        plt.plot(tarray,Smean)
        plt.xlabel('t')
        plt.ylabel('Smean')
        plt.title('Smean vs timestep')
        plt.show()

        plt.figure()
        plt.plot(tarray,Svar)
        plt.xlabel('t')
        plt.ylabel('Svar')
        plt.title('Svar vs timestep')
        plt.show()

    return Smean,Svar

def lineardiffusion(G,x=0,D=0.01,tf=6,Nt=400):
    """ This function is uded in the investigation for Part Question 3"""
    nodes=G.nodes()
    N=len(nodes)
    tarray = np.linspace(0,tf,Nt+1)

    L=nx.laplacian_matrix(G)
    L=L.todense()

    #Set up vector y to input in RHS
    y=np.zeros(3*N)
    y[2*N:]=1
    y[x],y[N+x],y[2*N+x]=0.05,0.05,0.1

    dy=np.zeros(3*N)

    def RHS_diffusion(y,t):
        dsdt=-D*np.matmul(L,y[:N])
        didt=-D*np.matmul(L,y[N:2*N])
        dvdt=-D*np.matmul(L,y[2*N:3*N])
        dy[:N]=dsdt
        dy[N:2*N]=didt
        dy[2*N:3*N]=dvdt

        return dy

    y=odeint(RHS_diffusion,y,tarray)

    return y

def infectionmodel(G,x=0,params=(50,80,105,71,1,0.01),tf=6,Nt=400):
    """
    This function is used in the investigation for Part 2 Question 3
    """

    a,theta0,theta1,g,k,tau=params
    tarray = np.linspace(0,tf,Nt+1)

    nodes=G.nodes()
    N=len(nodes)
    A=nx.adjacency_matrix(G)
    A=A.todense()

    q=A.sum(axis=0,dtype=float).tolist()[0]

    F=np.transpose(np.multiply(q,A))
    F=tau*np.divide(F,F.sum(axis=0))

    y=np.zeros(3*N)
    y[2*N:]=1
    y[x],y[N+x],y[2*N+x]=0.05,0.05,0.1

    dy=np.zeros(3*N)
    Fsum=np.sum(F,axis=0)

    def RHS1(y,t):
        """Compute RHS of model at time t
        input: y should be a 3N x 1 array containing with
        y[:N],y[N:2*N],y[2*N:3*N] corresponding to
        S on nodes 0 to N-1, I on nodes 0 to N-1, and
        V on nodes 0 to N-1, respectively.
        output: dy: also a 3N x 1 array corresponding to dy/dt

        Discussion: add discussion here
        """
        theta=theta0+theta1*(1-np.sin(2*np.pi*t))
        sum2=tau*y
        thetaSV=theta*np.multiply(y[:N],y[2*N:3*N])

        dsdt= a*y[N:2*N]-(g+k)*y[0:N]+np.matmul(F,y[:N])-sum2[:N]
        dy[:N]=dsdt

        didt=thetaSV-(k+a)*y[N:2*N]+np.matmul(F,y[N:2*N])-sum2[N:2*N]
        dy[N:2*N]=didt

        dvdt=k*(1-y[2*N:3*N])-thetaSV+np.matmul(F,y[2*N:3*N])-sum2[2*N:3*N]
        dy[2*N:3*N]=dvdt

        return dy

    y=odeint(RHS1,y,tarray)

    return y


def diffusion(figurenum,display=True):
    """Analyze similarities and differences
    between simplified infection model and linear diffusion on
    Barabasi-Albert networks.
    Modify input and output as needed.

    Discussion: add discussion here

    To investigate the similarities and differences between the resulting dynamics
    and linear diffusion on Barabasi-Albert graphs, I focused my investigation on
    the change in mean, change in variance and the effect of changing the parameters
    theta0 and tau.

    Changes in the mean:
    I first investigated into the effects that each model has on the mean. Figure 1
    is a plot that considers the infection model and the mean across all nodes in
    each state: Spreaders, Infectors, Vulnerable using the standard parameters theta0=80,
    tau=0.01. This plot shows that as the time increases the mean for the Spreaders
    remains constant throughout. (This can be explained by considering the equation
    for ds/dt. Taking the average over S is equivalent to summing over all of the nodes
    and dividing by N:

    Mean = (1/N )*( sum over i (sum over j (F_ij*S_j))- sum over I (sum over j (F_ji*S_i))=(1/N) *0 = 0)

    The plot also shows that as time increases the mean for the Infected state
    increases steadily and plateau’s approaching to a value 1.0 at roughly time = 350
    and remains at this value as time increases. On the other hand, the Vulnerable state
    shows the opposite trend and starts with a mean of 1.0 and decreases steadily and
    approaches and settles to 0 at roughly time = 350. This happens because the cells
    in the Vulnerable state become infected and transfer to the Infected state. The
    decrease in the mean for the Vulnerable state is proportional to the increase in
    mean for the Infected State, this can be seen from the equations for di/dt which
    has a theta*S*V term and dv/dt which has a -theta*S*V term.

    Figure 2 is a plot that considers the linear diffusion model and the mean across
    the nodes for each state: Spreaders, Infectors, Vulnerable. The plot shows that as
    time increases the mean for the Spreaders stays constant at 0 similar to the behaviour
    the mean of Spreaders in the original infection model. The mean for the Infected state
    also remains constant at 0 and the mean for the Vulnerable state remains constant at 1
    which is the same as the initial conditions for the N-1 nodes that are initially not infected.
    This is because the linear diffusion model has no interactions between the states S, I, V,
    whereas in the original infection model there are interactions between S, I, V which account
    for the different behaviours in the means between the two models.

    Changes in the variance:
    Next, I investigated into the effects that each model has on the variance. Figure 4
    is a plot that considers the linear diffusion model and the variance across all nodes
    in each state: Spreader, Infectors, Vulnerable using the diffusivity constant as D=0.01.
    This plot shows that for the Vulnerable state there is initially a high variance but
    this is due to the contribution of one initially infected node having an initial
    condition V=1. As time increases the variance quickly decreases for this state as
    diffusion occurs and an equilibrium is reached where the concentration is the same
    across all nodes. This is the same for the other states but on a smaller scale.

    Figure 3, is a plot which considers the original infection model and the variance
    across all nodes in  each state. For the Infected state the variance steadily increases
    and then starts to plateau as t is increases to 600. This can be explained by the trends
    in the mean across all nodes exhibited in Figure 1.  The mean for the Infected state
    increases and as it does the spread between the proportion of cells that are infected
    in each node increases. This causes the variance to increase. But as all of the vulnerable
    cells become infected the model reaches an equilibrium so the variance stabilizes and plateau’s.

    Unlike in the linear diffusion model, in the infection model the variance for the
    vulnerable state increases to a small local maximum and decreases down to almost zero.
    This is where all of the cells become infected and there are no vulnerable cells left
    hence the variance remains constant.  For both models the spreaders have a constant
    variance and this Is because the spreaders never convert to another state, they only
    travel between nodes so they number of cells that are spreaders remain constant and
    the variance is constant.

    Effects of theta0:
    The parameter theta0 only affects the original infection model but not the linear
    diffusion model. I have done a brief investigation into what ways theta0 affects the
    original infection model whilst setting tau=0.01 and below is a summary of my findings.

    Theta0 has no effect on the mean across all nodes for the Spreaders and the mean
    remains constant and flat throughout. I have not included this plot in my code. However,
    figure 5 shows that for the Infected state as theta0 is increases the mean across all
    nodes converges at a faster rate. This is because theta controls the conversion of
    vulnerable cells into infected cells, so these cells convert at a faster rate resulting
    in a higher mean. Likewise as the Vulnerable state has a negative theta term in it’s
    equation (-theta*S*V) figure 6 shows that for this state as theta0 is increased the
    mean across the nodes decrease and stabilize to 0 at a faster rate.

    Effects of tau:
    In the infection model the parameter tau acts as a diffusivity constant or constant
    that determines the convergence. For a direct comparison in the linear diffusion model
    I have set the diffusivity constant D to be the same as tau and I have varied the value
    of tau to see the effects on both models.

    For the infection models varying tau has no effect on the mean for the Spreaders
    state as Smean is always constant because of the reasons explained earlier.
    I have not included this plot in my code. However, Figure 7 and Figure 8 show that
    as tau increases the mean increases and converges for the infected state, likewise
    as tau increases the mean decreases and converges to 0 faster. Varying tau has
    some interesting effects on the variance for the Vulnerable state and the Infected State.
    Figure 9 shows that as tau is increased the variance for state V increases to a higher
    local maximum and then the variance decreases/ decays at a faster rate before plateauing.
    This is because the higher the tau the system reaches and equilibrium faster.

    """

    if figurenum==1:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=600,400
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)
        yinfection=infectionmodel(G,x=0,params=(0,80,0,0,0,0.01),tf=600,Nt=400)
        Smeaninfection=np.mean(yinfection[:,:N],axis=1)
        Imeaninfection=np.mean(yinfection[:,N:2*N],axis=1)
        Vmeaninfection=np.mean(yinfection[:,2*N:],axis=1)

        plt.figure()
        plt.hold(True)
        plt.plot(tarray,Smeaninfection,'-r',label='Spreaders')
        plt.plot(tarray,Imeaninfection,'-g',label='Infected')
        plt.plot(tarray,Vmeaninfection,'-b',label='Vulnerable')
        plt.plot(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mean across all nodes')
        plt.title('Manlin Chawla: diffusion(1) \n Infection Model: Mean values across all nodes in each state \n theta0=80,tau=0.01')
        if display==True:
            plt.show()

    if figurenum==2:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=600,400
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)
        ylinear=lineardiffusion(G,x=0,D=0.01,tf=600,Nt=400)
        Smeanlinear=np.mean(ylinear[:,:N],axis=1)
        Imeanlinear=np.mean(ylinear[:,N:2*N],axis=1)
        Vmeanlinear=np.mean(ylinear[:,2*N:],axis=1)

        plt.figure()
        plt.hold(True)
        plt.plot(tarray,Smeanlinear,'-r',label='Spreaders')
        plt.plot(tarray,Imeanlinear,'-g',label='Infected')
        plt.plot(tarray,Vmeanlinear,'-b',label='Vulnerable')
        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mean across all nodes')
        plt.title('Manlin Chawla: diffusion(2) \n Linear Diffusion Model: Mean values across all nodes in each state \n D=0.01')
        if display==True:
            plt.show()


    if figurenum==3:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=600,400
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)
        yinfection=infectionmodel(G,x=0,params=(0,80,0,0,0,0.01),tf=600,Nt=400)
        Svarinfection=np.var(yinfection[:,:N],axis=1)
        Ivarinfection=np.var(yinfection[:,N:2*N],axis=1)
        Vvarinfection=np.var(yinfection[:,2*N:],axis=1)

        plt.figure()
        plt.hold(True)
        plt.plot(tarray,Svarinfection,'-r',label='Spreaders')
        plt.plot(tarray,Ivarinfection,'-g',label='Infected')
        plt.plot(tarray,Vvarinfection,'-b',label='Vulnerable')
        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Variance across all nodes')
        plt.title('Manlin Chawla: diffusion(3) \n Infection Model:Variance across all nodes in each state \n theta0=80, tau=0.01')
        if display==True:
            plt.show()

    if figurenum==4:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=80,400
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)
        ylinear=lineardiffusion(G,x=0,D=0.01,tf=80,Nt=400)
        Svarlinear=np.var(ylinear[:,:N],axis=1)
        Ivarlinear=np.var(ylinear[:,N:2*N], axis=1)
        Vvarlinear=np.var(ylinear[:,2*N:3*N], axis=1)

        plt.figure()
        plt.hold(True)
        plt.plot(tarray,Svarlinear,'-r',label='Spreaders')
        plt.plot(tarray,Ivarlinear,'-g',label='Infected')
        plt.plot(tarray,Vvarlinear,'-b',label='Vulnerable')
        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Variance across all nodes')
        plt.title('Manlin Chawla: diffusion(4) \n Linear Diffusion Model:Variance across all nodes in each state \n D=0.01')
        if display==True:
            plt.show()

    if figurenum==5:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=600,400
        thetavalues=np.linspace(0,100,6)
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for theta0 in thetavalues:
            yinfection=infectionmodel(G,x=0,params=(0,theta0,0,0,0,0.01),tf=600,Nt=400)
            Imean=np.mean(yinfection[:,N:2*N],axis=1)
            plt.plot(tarray,Imean,label='theta0='+str(theta0))

        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mean across all nodes')
        plt.title('Manlin Chawla: diffusion(5) \n Infection Model:Mean across nodes in the state:Infectors for different values of theta \n tau=0.01')
        if display==True:
            plt.show()

    if figurenum==6:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=600,400
        thetavalues=np.linspace(0,100,6)
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for theta0 in thetavalues:
            yinfection=infectionmodel(G,x=0,params=(0,theta0,0,0,0,0.01),tf=600,Nt=400)
            Vmean=np.mean(yinfection[:,2*N:],axis=1)
            plt.plot(tarray,Vmean,label='theta0='+str(theta0))


        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mean across all nodes in the state')
        plt.title('Manlin Chawla: diffusion(6) \n Infection Model:Mean across nodes in the state:Vulnerable for different values of theta \n tau=0.01')
        if display==True:
            plt.show()

    if figurenum==7:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=600,400
        tauvalues=np.linspace(0,0.2,6)
        tauvalues[0]=0.01
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for tau in tauvalues:
            yinfection=infectionmodel(G,x=0,params=(0,80,0,0,0,tau),tf=600,Nt=400)
            Imean=np.mean(yinfection[:,N:2*N],axis=1)
            plt.plot(tarray,Imean,label='tau='+str(tau))

        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mean across all nodes in the state')
        plt.title('Manlin Chawla: diffusion(7) \n Infection Model:Mean across nodes in the state:Infectors for different values of tau \n theta0=80')
        if display==True:
            plt.show()

    if figurenum==8:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=600,400

        tauvalues=np.linspace(0,0.2,6)
        tauvalues[0]=0.01
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for tau in tauvalues:
            yinfection=infectionmodel(G,x=0,params=(0,80,0,0,0,tau),tf=600,Nt=400)
            Vmean=np.mean(yinfection[:,2*N:],axis=1)
            plt.plot(tarray,Vmean,label='tau='+str(tau))


        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mean across all nodes in the state')
        plt.title('Manlin Chawla: diffusion(8) \n Infection Model:Mean across nodes in the state:Vulnerable for different values of tau \n theta0=80')
        if display==True:
            plt.show()

    if figurenum==9:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=300,400
        tauvalues=np.linspace(0,0.2,6)
        tauvalues[0]=0.01
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for tau in tauvalues:
            yinfection=infectionmodel(G,x=0,params=(0,80,0,0,0,tau),tf=300,Nt=400)
            Vvar=np.var(yinfection[:,2*N:],axis=1)
            plt.plot(tarray,Vvar,label='tau='+str(tau))


        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Variance across all nodes in the state')
        plt.title('Manlin Chawla: diffusion(9) \n Infection Model:Variance across nodes in the state:Vulnerable for different values of tau \n theta0=80')
        if display==True:
            plt.show()

    if figurenum==10:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=600,400
        tauvalues=np.linspace(0,0.2,6)
        tauvalues[0]=0.01
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for tau in tauvalues:
            ylinear=lineardiffusion(G,x=0,D=tau,tf=600,Nt=400)
            Imean=np.mean(ylinear[:,N:2*N],axis=1)
            plt.plot(tarray,Imean,label='D='+str(tau))

        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mean across all nodes in the state')
        plt.ylim((0,0.001))
        plt.title('Manlin Chawla: diffusion(10) \n Linear Diffusion Model:Mean across nodes in the state:Infectors for different values of D')
        if display==True:
            plt.show()


    if figurenum==11:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=40,400
        tauvalues=np.linspace(0,0.2,6)
        tauvalues[0]=0.01
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for tau in tauvalues:
            ylinear=lineardiffusion(G,D=tau,tf=40,Nt=400)
            Svar=np.var(ylinear[:,:N], axis=1)
            plt.plot(tarray,Svar,label='tau='+str(tau))

        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Variance across all nodes in the state')
        plt.title('Manlin Chawla: diffusion(11) \n Linear Diffusion Model:Variance across nodes in the state:Spreaders for different values of tau \n tau=0.01')

        if display==True:
            plt.show()

    if figurenum==12:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=40,400
        tauvalues=np.linspace(0,0.2,6)
        tauvalues[0]=0.01
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for tau in tauvalues:
            ylinear=lineardiffusion(G,D=tau,tf=40,Nt=400)
            Ivar=np.var(ylinear[:,N:2*N],axis=1)
            plt.plot(tarray,Ivar,label='tau='+str(tau))


        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Variance across all nodes in the state')
        plt.title('Manlin Chawla: diffusion(12) \n Linear Diffusion Model:Variance across nodes in the state:Infectors for different values of tau \n tau=0.01')

        if display==True:
            plt.show()


    if figurenum==13:
        G=nx.barabasi_albert_graph(100,5)
        tf,Nt=40,400
        tauvalues=np.linspace(0,0.2,6)
        tauvalues[0]=0.01
        N=nx.number_of_nodes(G)
        tarray = np.linspace(0,tf,Nt+1)

        plt.figure()
        plt.hold(True)

        for tau in tauvalues:
            ylinear=lineardiffusion(G,D=tau,tf=40,Nt=400)
            Vvar=np.var(ylinear[:,2*N:],axis=1)
            plt.plot(tarray,Vvar,label='tau='+str(tau))

        plt.hold(False)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Variance across all nodes in the state')
        plt.title('Manlin Chawla: diffusion(13) \n Linear Diffusion Model:Variance across nodes in the state:Vulnerable for different values of tau \n tau=0.01')

        if display==True:
            plt.show()

        return None #modify as needed


if __name__=='__main__':
    G=nx.barabasi_albert_graph(100,5)
    output_fig = diffusion(1,display=False)
    plt.savefig('fig1.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(1) figure saved!')

    output_fig = diffusion(2,display=False)
    plt.savefig('fig2.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(2) figure saved!')

    output_fig = diffusion(3,display=False)
    plt.savefig('fig3.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(3) figure saved!')

    output_fig = diffusion(4,display=False)
    plt.savefig('fig4.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(4) figure saved!')

    output_fig = diffusion(5,display=False)
    plt.savefig('fig5.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(5) figure saved!')

    output_fig = diffusion(6,display=False)
    plt.savefig('fig6.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(6) figure saved!')

    output_fig = diffusion(7,display=False)
    plt.savefig('fig7.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(7) figure saved!')

    output_fig = diffusion(8,display=False)
    plt.savefig('fig8.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(8) figure saved!')

    output_fig = diffusion(9,display=False)
    plt.savefig('fig9.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(9) figure saved!')

    output_fig = diffusion(10,display=False)
    plt.savefig('fig10.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(10) figure saved!')

    output_fig = diffusion(11,display=False)
    plt.savefig('fig11.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(11) figure saved!')

    output_fig = diffusion(12,display=False)
    plt.savefig('fig12.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(12) figure saved!')

    output_fig = diffusion(13,display=False)
    plt.savefig('fig13.png', bbox_inches="tight")
    plt.clf()
    print('diffusion(13) figure saved!')
