import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from My_Classes import mc


K=5 # Number of trajectories
M=50 # Number of Fourier terms to use
N=50 #
num = 100 #num = nx = nx, number of points representing x and y 
nx = num
ny = num
nt = 25
a,b=1,1
dx = 1/(nx-1)
dy = 1/(ny-1)
dt = .01
TF = nt*dt
c=1
k=1
sp=.8
SP = int(K*sp) 

x = np.linspace(0,a,num,endpoint=True)
y = x.copy()#Square domain

t = np.linspace(0,TF,nt,endpoint=True)
xx,yy = np.meshgrid(x,y)
zz = np.sin(np.pi*xx/a)*np.sin(np.pi*yy/b)
def st(a):
    return (a-a.mean())/(a.std())

def p2(a,b,x=xx,y=yy):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.plot_surface(x,y,a,alpha=.5,cmap=cm.viridis)
    ax.scatter(x,y,b,alpha=.8,label='DNN Prediction',c='orange')
    plt.legend()
    plt.show()

def plot_movie(U):
    fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
    X,Y = np.meshgrid(x,y)
    preim = []
    ims = []

    for i in range(len(U)):
        ax.axes.set_zlim3d(bottom=-.7,top=.7)
        ui = U[int(i)]
        im = ax.plot_surface(X,Y,ui,animated=True,cmap=plt.get_cmap('rainbow'),alpha=.7)
        ims.append([im])
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
    plt.show()

    
def random_surface():
    def intr_bdry(xx=xx,yy=yy): #Set the coordinates up in a 1D list of tuples
        bottom = np.concatenate((xx[0][:,None],yy[0][:,None]),axis=1)
        right = np.concatenate((xx[:,-1][:,None],yy[:,-1][:,None]),axis=1)
        left = np.concatenate((xx[:,0][:,None],yy[:,0][:,None]),axis=1)
        top = np.concatenate((xx[-1][:,None],yy[-1][:,None]),axis=1)

        e=7 #Don't want any large values near the boundary, so chop e off of edges
        intr_locs = np.concatenate((np.ravel(xx[e:-e,e:-e])[:,None],np.ravel(yy[e:-e,e:-e])[:,None]),axis=1)
        bdry_locs = np.concatenate((bottom,right,left,top),axis=0)
        bdry_vals = np.zeros_like(bdry_locs[:,0])
        return intr_locs, bdry_locs, bdry_vals

    intr_locs, bdry_locs, bdry_vals = intr_bdry()

    def bumps(intr_locs=intr_locs):
        some_intr = intr_locs
        some_inds = np.arange(0,len(some_intr),1)
        siz = 14
        random_inds = np.random.choice(some_inds,size=siz,replace=False)
        
        bump_locs = some_intr[random_inds] #Hmm, why keep intr_locs then?
        bump_vals = np.zeros_like(bump_locs[:,0]) 
        
        for i in range(len(bump_vals)):
            bump_vals[i] = np.random.normal(0,1)
        return bump_locs, (bump_vals-bump_vals.mean())/(bump_vals.std()) #Random points

    def support_vector(boL,buL,boV,buV): #Fit a surface to the random points (bumps) and bdry
        
        X_ = np.concatenate((boL,buL),axis=0)
        buV_unit = buV/np.linalg.norm(buV)
        Z_ = np.concatenate((boV,buV_unit),axis=0)

        model = SVR(degree=90000,C=50000.) #We really want to fasten the bdry at 0...
        model.fit(X_,Z_)

        IC = model.predict(np.c_[xx.ravel(),yy.ravel()]) #The surface will act as the IC
        IC = IC.reshape(xx.shape)                       
        return IC
    rand_surf = support_vector(bdry_locs,bumps()[0],bdry_vals,bumps()[1])
    return zz*rand_surf

rand_surf = random_surface()




def fourier_solution(z=rand_surf,x=x,y=y):

    def sin(x,m): #X(x)=phi, Y(y)=psi
        return np.sin(x*np.pi*m)
    def ksi(t,m,n): #T(t) from the separated variable u=X(x)Y(y)T(t) equation
        return np.cos(k*np.pi*np.sqrt((m/a)**2+(n/b)**2)*t)
    def bmn(z,m,n): #Fourier coefficients on the cosine term
        phi = sin(x,m)
        psi = sin(y,n)
        Pxx,Pyy = np.meshgrid(phi,psi)
        return z*Pxx*Pyy

    def Bmn(z,M=M,N=N): #The full matrix of all the bmn coefficients
        B = np.zeros((M,N))
        for i in range(M):
            phi = sin(x,i)
            for j in range(N):
                psi = sin(y,j)
                B[i,j]= 4/(a*b)*np.sum(bmn(z,i,j))*dx*dy
        return B

    B = Bmn(z=z)
    
    def func_mn_tau(i=1,j=1,tau=0,ks=True): #Corresponding to every choice (m,n) there is
        phi = sin(x,i)              #a surface at a fixed time tau. They are summed
        psi = sin(y,j)              #in the Fourier series.
        px,py = np.meshgrid(phi,psi)
        T = tau*dt*np.ones_like(px)
        if ks==True:
            return B[i,j]*px*py*ksi(T,i,j) #This represents one such surface corresp. to (m,n)=(i,j)
        else:
            return B[i,j]*px*py
        
    def solution_operator(tau=0, all_=True, M=M, N=N,ks=True): 
        f = np.array([[func_mn_tau(i,j,tau,ks=ks) for i in range(M)] for j in range(N)])
        fij = np.zeros_like(f[0][0]) #sum all of the M*N functions in the Fourier series
        for i in range(N):
            for j in range(M):
                fij+=f[i][j]
        return fij

    def trajectory(): #This returns an array of surfaces, thus subscripting the array
        surfaces = []                               # acts as though we are calling
        for i in range(nt):                         # a solution operator [from IDDS]
            surface = solution_operator(tau=i)                   # at time i/nt
            surfaces.append(surface)
        return surfaces
    
    return np.array(trajectory())

sol = fourier_solution(rand_surf)
mc.dot_plot(sol)
##
##def many_sols(K=K):
##    S_ = []
##    for i in range(K):
##        ri = random_surface()
##        si = fourier_solution(z=ri)
##        S_.append(si)
##    S = np.array(S_)
##    return S
##
##def dom(x,y,t,K=K):
##    t = t[0:-1]
##    xx,yy = np.meshgrid(x,y)
##    x_ = np.ravel(xx)[:,None]
##    y_ = np.ravel(yy)[:,None]
##    t_ = np.ones_like(t)[:,None]
##    t_ = np.ones_like(x_)
##    X_ = np.concatenate((x_,y_),axis=1)
##    SX_ = []
##    T_ = []
##    for i in range(nt-1):
##        SX_.append(X_)
##        T_.append(t[i]*t_)
##    SX_ = np.array(SX_)
##    SX_ = SX_.reshape((SX_.shape[0]*SX_.shape[1] , SX_.shape[2]))
##    T_ = np.array(T_)
##    T_ = T_.reshape((T_.shape[0]*T_.shape[1] , T_.shape[2]))
##    
##    D = np.concatenate((SX_, T_),axis=1)
##    all_D = []
##    for i in range(K):
##        all_D.append(D)
##
##    All_D = np.array(all_D)
##    return All_D.reshape((K*All_D.shape[1],All_D.shape[2]))
##
##def Arrange_Data(K=K):
##    trajs_0 = []
##    trajs_1 = []
##    S = many_sols()
##    
##    for i in range(K):
##        trajs_1.append(S[i][1:])
##        trajs_0.append(S[i][:-1])
##    Y_G_ = np.array(trajs_1).reshape((K*ny*nx*(nt-1),1))
##    X_U_ = np.array(trajs_0).reshape((K*ny*nx*(nt-1),1))
##    return Y_G_,X_U_
##
##
##def DNN(K=K):
##    import deepxde as dde
##    
##    YGTrain,XUTrain = Arrange_Data(K)
##    XDTrain = dom(x,y,t,K)
##
##    YGTest,XUTest = Arrange_Data(int(K/4))
##    XDTest = dom(x,y,t,int(K/4))
##    
##    data = dde.data.Triple((XUTrain,XDTrain),YGTrain,(XUTest,XDTest),YGTest)
##
##    net = dde.nn.DeepONet([1,20,400,200,100],
##               [3,20,400,200,100],
##               activation='relu',
##               kernel_initializer='Glorot normal',
##               use_bias=True,
##               stacked=False,
##               trainable_trunk=True)
##
##    
##    model = dde.Model(data,net)
##    model.compile('adam',lr=.007)
##    lh,ts = model.train(epochs=5000,display_every=500,batch_size=50)
##    return lh,ts,model
##
##
##lh,ts,model = DNN()
##pred = ts.best_y[:(nt-1)*num*num].reshape((nt-1,num,num))
##plot_movie(pred)
##test = ts.y_test[:(nt-1)*num*num].reshape((nt-1,num,num))
##
##def compute_loss(ts):
##    Pred = ts.best_y
##    Test = ts.X_test[0]
##
##    lp = len(Pred)
##    siz = int((nt-1)*nx*ny)
##    k = int(lp/siz)
##
##    Preds = Pred.reshape((k,nt-1,ny,nx))
##    Tests = Test.reshape((k,nt-1,ny,nx))
##    
##    dists = np.zeros(nt-1)
##    
##    for i in range(k):
##        d = np.zeros(nt-1)
##        for n in range(nt-1):
##            d[n] = np.linalg.norm(Preds[i][n]-Tests[i][n])
##        dists = dists + d
##    return dists/k
##
##loss = compute_loss(ts)
##
##def plot_loss(loss):
##    fig = plt.figure()
##    ax = fig.add_subplot()
##    ax.set_xlabel('t')
##    ax.set_ylabel('loss')
##    ax.plot(t[1:],loss,label='|True(t)-DNN(t)|',lw=3,c='black')
##    plt.legend()
##    plt.show()
##
##plot_loss(loss)
##





