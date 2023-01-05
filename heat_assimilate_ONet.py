import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from My_Classes import mc

K=50 # Number of trajectories
num = 30 #num = nx = nx, number of points representing x and y 
nx = num
ny = num
nt = 150

start_nt = 0
end_nt = nt
indices_nt = nt-start_nt

a,b=1,1
dx = 1/(nx-1)
dy = 1/(ny-1)
dt = .002
TF = nt*dt
c=1
k=1
sp=.8
SP = int(K*sp)
alpha=.1

x = np.linspace(0,a,num,endpoint=True)
y = x.copy()#Square domain

t = np.linspace(0,TF,nt,endpoint=True)
t_chopped = t[start_nt:end_nt]


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

def plot_movies(U,V,G,lab='temp'):
    fig,ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(11,11))
    X,Y = np.meshgrid(x,y)
    preim = []
    ims = []

    for i in range(len(U)):
        ax.axes.set_zlim3d(bottom=-.3,top=.3)
        ax.set_axis_off()
        ui = U[int(i)]
        vi = V[int(i)]
        gi = G[int(i)]

        im = ax.plot_surface(X,Y,ui,animated=True,cmap=plt.get_cmap('rainbow'),alpha=.5)
        imm = ax.scatter(X,Y,vi,animated=True,alpha=.9,c='green',s=25)
        immm = ax.scatter(X,Y,gi,animated=True,alpha=.9,c='orange',s=30,marker='$X$')
        ims.append([im,imm,immm])
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)

    #ani.save('a.gif',writer='imagemagick',fps=10)
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

        model = SVR(degree=900,C=500.) #We really want to fasten the bdry at 0...
        model.fit(X_,Z_)

        IC = model.predict(np.c_[xx.ravel(),yy.ravel()]) #The surface will act as the IC
        IC = IC.reshape(xx.shape)                       
        return IC
    rand_surf = support_vector(bdry_locs,bumps()[0],bdry_vals,bumps()[1])
    return zz*rand_surf

rand_surf = random_surface()
rand_surf_2 = random_surface()
#plt.contourf(rand_surf,cmap=cm.rainbow,levels=np.linspace(-.5,.5,1000))
#plt.show()



def FDM(un=rand_surf,nt=nt):
    u = un
    U = np.zeros((nt,ny,nx))


    for n in range(nt):
        un = u.copy()
        u[1:-1, 1:-1] = alpha*dt*(1/dx**2 *(un[0:-2, 1:-1]-2*un[1:-1,1:-1]+un[2:, 1:-1] )+1/dy**2 *(un[1:-1, 0:-2]-2*un[1:-1, 1:-1] + un[1:-1, 2:])) + un[1:-1,1:-1]
        
        u[:,0]=0
        u[:,-1]=0
        u[0,:]=0
        u[-1,:]=0

        U[n] = u
    return U

U = FDM()

def scat(U):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(bottom=-.35,top=.35)
    ax.set_axis_off()
    ax.view_init(azim=15,elev=15)
    ax.plot_surface(xx,yy,U,alpha=.5,cmap=cm.rainbow)
    ax.scatter(xx,yy,U,c='blue',alpha=.7,s=10)
    plt.show()

##scat(U[0])
##scat(U[25])
##scat(U[50])
##scat(U[-1])
mu = 30
def nudge(wn, U=U):
    w = wn
    W = np.zeros((nt,ny,nx))

    for n in range(nt):
        wn = w.copy()
        w[1:-1,1:-1] = dt*(\
            alpha*(1/dx**2 *(wn[0:-2, 1:-1]-2*wn[1:-1,1:-1]+wn[2:, 1:-1] )+1/dy**2 *(wn[1:-1, 0:-2]-2*wn[1:-1, 1:-1] + wn[1:-1, 2:]))\
            - mu*(wn[1:-1,1:-1]-U[n][1:-1,1:-1])) + wn[1:-1,1:-1]

        w[:,0]=0
        w[:,-1]=0
        w[0,:]=0
        w[-1,:]=0

        W[n] = w
    return W

W = nudge(random_surface())


def many_sols(K=K,Nudge=True):
    S_ = []
    NS_ = []
    
    rands = [random_surface() for i in range(K)]
    if(Nudge==True):
        rands_n = [random_surface() for i in range(K)]
    
    for i in range(K):
        ri = rands[i]
        si = FDM(un=ri)
        S_.append(si[start_nt:end_nt]) #chopping early times off
        
        if(Nudge==True):
            rii = rands_n[i]
            nsi = nudge(rii,si)
            NS_.append(nsi[start_nt:end_nt])
        
    S = np.array(S_)
    if(Nudge==True):
        N_ = np.array(NS_)
        return S,N_
    else:
        return S


def dom(x,y,t=t_chopped,K=K,nt=indices_nt):
    t = t[0:-1] #make sure to use (input) the chopped time interval
    xx,yy = np.meshgrid(x,y)
    x_ = np.ravel(xx)[:,None]
    y_ = np.ravel(yy)[:,None]
    t_ = np.ones_like(t)[:,None]
    t_ = np.ones_like(x_)
    X_ = np.concatenate((x_,y_),axis=1)
    SX_ = []
    T_ = []
    for i in range(nt-1):
        SX_.append(X_)
        T_.append(t[i]*t_)
    SX_ = np.array(SX_)
    SX_ = SX_.reshape((SX_.shape[0]*SX_.shape[1] , SX_.shape[2]))
    T_ = np.array(T_)
    T_ = T_.reshape((T_.shape[0]*T_.shape[1] , T_.shape[2]))
    
    D = np.concatenate((SX_, T_),axis=1)
    all_D = []
    for i in range(K):
        all_D.append(D)

    All_D = np.array(all_D)
    return All_D.reshape((K*All_D.shape[1],All_D.shape[2]))

def Arrange_Data(K=K,nt=indices_nt):
    trajs_0 = []
    trajs_1 = []

    N_trajs_0 = []
    N_trajs_1 = []
    
    S,N_ = many_sols()
    
    for i in range(K):
        trajs_1.append(S[i][1:])
        trajs_0.append(S[i][:-1])

        N_trajs_1.append(N_[i][1:])
        N_trajs_0.append(N_[i][:-1])
        
    Y_G_ = np.array(N_trajs_1).reshape((K*ny*nx*(nt-1),1))
    X_U_ = np.array(N_trajs_0).reshape((K*ny*nx*(nt-1),1))
    X_D_ = np.array(trajs_0).reshape((K*ny*nx*(nt-1),1))
    #X_U_ = np.array(trajs_0).reshape((K*ny*nx*(nt-1),1))
    
    return Y_G_,X_U_,X_D_


def DNN(t,K=K): #make sure to use (input) the chopped time interval
    import deepxde as dde

    YGTrain,XUTrain,XDTrain = Arrange_Data(K)
    
    #YGTrain,XUTrain = Arrange_Data(K)
    #XDTrain = dom(x,y,t,K)

    YGTest,XUTest,XDTest = Arrange_Data(int(K/4))
    #XDTest = dom(x,y,t,int(K/4))
    
    data = dde.data.Triple((XUTrain,XDTrain),YGTrain,(XUTest,XDTest),YGTest)

    net = dde.nn.DeepONet([1,20,400,200,100],
               [1,20,400,200,100],
               activation='relu',
               kernel_initializer='Glorot normal',
               use_bias=True,
               stacked=False,
               trainable_trunk=True)

    
    model = dde.Model(data,net)
    model.compile('adam',lr=.007)
    lh,ts = model.train(epochs=5000,display_every=500,batch_size=20)
    return lh,ts,model


lh,ts,model = DNN(t_chopped)
#def trials(lef=0,rig=1):
#    pred = ts.best_y[(indices_nt-1)*num*num*lef:rig*(indices_nt-1)*num*num].reshape((rig-lef)*((indices_nt-1),num,num))
#    test = ts.y_test[(indices_nt-1)*num*num*lef:rig*(indices_nt-1)*num*num].reshape((rig-lef)*((indices_nt-1),num,num))
#    true = ts.X_test[1][(indices_nt-1)*num*num*lef:rig*(indices_nt-1)*num*num].reshape((rig-lef)*((indices_nt-1),num,num))
#    return true,test,pred

#true,test,pred = trials(7,8)
#true= ts.best_y[:(indices_nt-1)*num*num].reshape((indices_nt-1,num,num))
pred = ts.best_y[0:(indices_nt-1)*num*num].reshape((indices_nt-1,num,num))
test = ts.y_test[0:(indices_nt-1)*num*num].reshape((indices_nt-1,num,num))
true = ts.X_test[1][:(indices_nt-1)*num*num].reshape((indices_nt-1,num,num))
#plot_movies(pred,test,true)
#plot_movies(true,test,pred)

#for i in range(3):
#    true,test,pred = trials(7+i,8+i)
#    plot_movies(true,test,pred,lab=str(i))

    
def compute_loss(ts,nt=indices_nt):
    Pred = ts.best_y
    Test = ts.X_test[0]
    True_ = ts.y_test

    lp = len(Pred)
    siz = int((nt-1)*nx*ny)
    k = int(lp/siz)

    Preds = Pred.reshape((k,nt-1,ny,nx))
    Tests = Test.reshape((k,nt-1,ny,nx))
    Trues_ = True_.reshape((k,nt-1,ny,nx))
    
    dists = np.zeros(nt-1)
    dists_ = dists.copy()
    dists__ = dists.copy()
    
    for i in range(k):
        d = np.zeros(nt-1)
        d_ = d.copy()
        d__ = d.copy()
        for n in range(nt-1):
            d[n] = np.linalg.norm(Preds[i][n]-Tests[i][n])
            d_[n] = np.linalg.norm(Tests[i][n]-Trues_[i][n])
            d__[n] = np.linalg.norm(Trues_[i][n]-Preds[i][n])
        dists = dists + d
        dists_ = dists_ + d_
        dists__ = dists__ + d__
        
    return dists/k, dists_/k, dists__/k

loss,loss_,loss__ = compute_loss(ts)

def plot_loss(loss,loss_,loss__):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('t')
    ax.set_ylabel('loss')
    ax.plot(t_chopped[1:],loss,label='|DNN(t)-Nudge(t)|',lw=3,c='cyan')
    ax.plot(t_chopped[1:],loss_,label='|Nudge(t)-True(t)|',lw=3,c='black')
    ax.scatter(t_chopped[1:],loss__,label='|True(t)-DNN(t)|',s=10,c='magenta')
    plt.legend()
    plt.show()

plot_loss(loss,loss_,loss__)

def plot_movies_new(U,V,G,lab='temp'):
    fig,ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(10,10))
    loss_fig, loss_ax = plt.subplots(figsize=(7,7))
    X,Y = np.meshgrid(x,y)

    preim = []
    ims = []
    loss_ims = []

    t_ = t[:-1]
    for i in range(len(U)):
        ax.axes.set_zlim3d(bottom=-.3,top=.3)
        ax.set_axis_off()
        ui = U[int(i)]
        vi = V[int(i)]
        gi = G[int(i)]

        loss_im = loss_ax.scatter(t_[:i],loss[:i],c='cyan')
        loss_im_ = loss_ax.scatter(t_[:i],loss_[:i],c='black')
        loss_im__ = loss_ax.scatter(t_[:i],loss__[:i],c='magenta')
        loss_ims.append([loss_im,loss_im_,loss_im__])


        im = ax.plot_surface(X,Y,ui,animated=True,cmap=plt.get_cmap('rainbow'),alpha=.5)
        imm = ax.scatter(X,Y,vi,animated=True,alpha=.9,c='green',s=25)
        immm = ax.scatter(X,Y,gi,animated=True,alpha=.9,c='orange',s=30,marker='$X$')
        ims.append([im,imm,immm])
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
    loss_ani = animation.ArtistAnimation(loss_fig,loss_ims,interval=50,blit=True)
    plt.show()
