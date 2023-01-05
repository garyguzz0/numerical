import numpy as np
from scipy import integrate
from numpy.random import uniform
from numpy.random import normal
from numpy.random import randint
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib import animation
import os
import copy
from My_Classes import mc
from My_Classes import Lorenz
from scipy.integrate import solve_ivp,odeint
from matplotlib import cm
from My_Classes import Lorenz



N = 3000
N+=1
C=100
NC = N-C-1
n=N-1

num = N-1-C
eta = .01
T = eta*N
t = np.linspace(C*eta,eta*(N-1),N-C-1)
t_ = t[1:]


L0 = np.array((0,normal(.1,1),normal(.2,1)))
W0 = np.array((0, .1, .2))

s = 10
b = 8/3
r = 28
r = s*1.0*(s*1.0+b*1.0+3*1.0)/(1.0*s-1.0*b-1.0)
r=22.2
r=28
nu = 30
K = 1000

params = [N,eta,T]

def Lfunc(t,X,r=28,b=b):
    x,y,z = X
    dx = s*(y-x)
    dy = r*x-y-x*z
    dz = x*y-b*z
    return np.array((dx,dy,dz))
    
#_________________#
sol = odeint(Lfunc,[30,30,5],np.arange(0,100,.01),tfirst=True,args=(28,b))
sol_ = odeint(Lfunc,[-30,5,1],np.arange(0,100,.01),tfirst=True,args=(22,b))

def ode_solve(ic, func=Lfunc, args=(28,b) ):
    return odeint(func,ic,np.arange(0,100,.01),tfirst=True,args=args)

v1=-1*np.sqrt(848.0/15.0)
v2 = 106.0/5.0

def plo(L,L_):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(L[:,0],L[:,1],L[:,2],c=np.linspace(0,1,len(L)),cmap=cm.rainbow)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(L_[:,0],L_[:,1],L_[:,2],c=np.linspace(0,1,len(L)),cmap=cm.magma,alpha=.25)
    #ax.scatter(v1,v1,v2,c='red',s=50,marker='D')
    plt.show()
plo(sol,sol_)


def Randy():
    r1 = normal(0,2)
    r2 = normal(0,2)
    r3 = normal(0,2)
    randy = np.array((r1,r2,r3))
    return randy

def RandNear(randy = Randy()):
    r1 = randy[0]+normal(randy[0],.00000005)
    r2 = randy[1]+normal(randy[1],.00000005)
    r3 = randy[2]+normal(randy[2],.00000005)
    randnear = np.array((r1,r2,r3))
    return randnear

def solve_2(randy=Randy()):
    L0s = []
    Ls = []
    L0 = randy
    for i in range(2):
        L0s.append(RandNear(L0))
        Ls.append(ode_solve(L0s[i]))
    #Ls = Ls[-2000:]
    return Ls
    


def plott():
    L1,L2 = solve_2()
    L0 = Randy()
    L3,L4 = solve_2(randy=L0)
    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    T_ = np.linspace(0,10,len(L1))
    ax.scatter(L1[:,0],L1[:,1],L1[:,2],c=T_,cmap=cm.rainbow,s=35,label=str(L1[0]),alpha=.3,marker='$X$')
    ax.scatter(L2[:,1],L2[:,1],L2[:,2],c=T_,cmap=cm.magma,s=15,label=str(L2[0]),alpha=.3,)
    ax.scatter(L3[:,1],L3[:,1],L3[:,2],c=T_,cmap=cm.viridis,s=35,label=str(L3[0]),alpha=.3,marker='$X$')
    ax.scatter(L4[:,1],L4[:,1],L4[:,2],c=T_,cmap=cm.spring,s=15,label=str(L4[0]),alpha=.3)

    ax.plot(np.linspace(-7.52-20,-7.52+20,100),-7.52*np.ones(100),21.2*np.ones(100),c='black')
    ax.plot(-7.52*np.ones(100),np.linspace(-7.52-20,-7.52+20,100),21.2*np.ones(100),c='black')
    ax.plot(-7.52*np.ones(100),-7.52*np.ones(100),np.linspace(21.2-14,21.2+14,100),c='black')

    ax.plot(np.linspace(7.52-20,7.52+20,100),7.52*np.ones(100),21.2*np.ones(100),c='black')
    ax.plot(7.52*np.ones(100),np.linspace(7.52-20,7.52+20,100),21.2*np.ones(100),c='black')
    ax.plot(7.52*np.ones(100),7.52*np.ones(100),np.linspace(21.2-14,21.2+14,100),c='black')

    

    #ax.scatter(-7.52,-7.52,21.2,c='black',marker='$X$',s=100)
    plt.legend()
    plt.show()
    return L1,L2,L3

L1,L2,L3 = plott()
    

def step(L,W,eta,extra=False):
    W_pre = W.copy()
    dx = s*(L[1]-L[0])
    dy = r*L[0]-L[1]-L[0]*L[2]
    dz = L[0]*L[1]-b*L[2]
    dL = np.array((dx,dy,dz))

    Dx_pre = s*(W[1]-W[0])
    nu_term = nu*(W[0]-L[0])
    
    Dx = Dx_pre - nu_term
    Dy = r*W[0]-W[1]-W[0]*W[2]
    Dz = W[0]*W[1]-b*W[2]
    DW = np.array((Dx,Dy,Dz))
    DW_pre = np.array((Dx_pre,Dy,Dz))

    if(extra==True):
        return L+eta*dL , W+eta*DW , W_pre+eta*DW_pre , nu_term
    else:
        return L+eta*dL, W+eta*DW

def DA_Lorenz(L_0=np.array((30,30,5)), W_0 = np.array((0,.1,.2)), N=N , nu=nu,eta=eta,extra=False): #n is number of timestamps, L_0, W_0 are ICs

    W_0 = np.array((0,.1,.2))
    
    lorenz = np.zeros((N,3))
    worenz = lorenz.copy()

    if(extra==True):
        pre_worenz = np.zeros((N,3))
        nu_terms = np.zeros((N))
    
    lorenz[0]=L_0
    worenz[0]=W_0
    
    for i in range(1,N):
        if(extra==True):
            lorenz[i],worenz[i],pre_worenz[i],nu_terms[i] = step(lorenz[i-1],worenz[i-1],eta=eta,extra=extra)
        else:
            lorenz[i],worenz[i] = step(lorenz[i-1],worenz[i-1],eta=eta,extra=extra)

    if(extra==True):
        return lorenz,worenz, pre_worenz, nu_terms, 
    else:
        return lorenz, worenz,


r1 = normal(0,10)
r2 = normal(0,10)
r3 = normal(0,10)
randy = np.array((r1,r2,r3))

def old_DA_Lorenz(L_0=randy,W_0 = np.array((0,.1,.1)), N=N, nu=nu,eta=eta):
    lorenz = np.zeros((N,3))
    worenz = lorenz.copy()

    lorenz[0] = L_0
    worenz[0] = W_0

    for i in range(1,N):
        lorenz[i],worenz[i] = step(lorenz[i-1],worenz[i-1],eta=eta)
    return lorenz,worenz


#L,W,PW,NU = DA_Lorenz(L_0 = randy, extra=True)
#L,W = DA_Lorenz(L_0 = randy)
L,W = old_DA_Lorenz(L_0=randy)


def plot3(L,W,PW,l=100,r=120):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #x.set_axis_off()
    ax.scatter(L[l:r,0],L[l:r,1],L[l:r,2],c='blue',s=np.linspace(15,45,len(L[l:r])))
    ax.scatter(W[l:r,0],W[l:r,1],W[l:r,2],c='red',s=np.linspace(15,45,len(L[l:r])))
    ax.scatter(PW[l:r,0],PW[l:r,1],PW[l:r,2],c='black',s=np.linspace(15,45,len(L[l:r])))#s=np.linspace(5,25,len(L))
    for i in range(len(L[l:r])):
        ax.plot(np.linspace(W[l:r][i,0],PW[l:r][i,0],5),PW[l:r][i,1]*np.ones((5)),PW[l:r][i,2]*np.ones((5)),c='brown',lw=2)
    ax.view_init(azim=-90,elev=20)
    plt.show()

#plot3(L,W,PW)        

#L,W = DA_Lorenz()
#L_,W_ = DA_Lorenz(L_0=np.array((0,-30,5)))

#plot2(L[:500],L_[:500])

def p1(L):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.plot(L[2000:,0],L[2000:,1],L[2000:,2],c='black')
    #plt.colorbar()
    plt.show()

#plot2(L[:500],W[:500])

#p1(L)



    

def plot_movie(L,W):
    fps=10
    fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
    loss_fig,loss_ax = plt.subplots()
    x_fig, x_ax = plt.subplots()
    
    ims = []
    loss_ims = []
    x_ims = []
    T_ = np.linspace(0,T,len(L))
    losses = np.array([np.linalg.norm(L[i]-W[i]) for i in range(len(L))])
    
    for i in range(len(L)):
        
        im = ax.scatter(L[i,0],L[i,1],L[i,2],c='blue')
        im = ax.scatter(L[i,0],L[i,1],L[i,2],c='blue')
        im_ = ax.scatter(L[max(0,i-100):i,0],L[max(0,i-100):i,1],L[max(0,i-100):i,2],c='blue',s=2,alpha=.2)
        imm = ax.scatter(W[i,0],W[i,1],W[i,2],c='red')
        imm_ = ax.scatter(W[max(0,i-100):i,0],W[max(0,i-100):i,1],W[max(0,i-100):i,2],c='red',s=2,alpha=.2)
        loss_im = loss_ax.scatter(T_[:i],losses[:i],c='black',s=5)
        x_im = x_ax.scatter(T_[:i],L[:i,0],c='blue',s=5)
        x_imm = x_ax.scatter(T_[:i],W[:i,0],c='red',s=5)
        ax.view_init(-140,60)
        ims.append([im,imm,im_,imm_])
        loss_ims.append([loss_im])
        x_ims.append([x_im,x_imm])
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
    loss_ani = animation.ArtistAnimation(loss_fig,loss_ims,interval=50,blit=True)
    x_ani = animation.ArtistAnimation(x_fig,x_ims,interval=500/fps,blit=True)
    

    fn = 'my_video'
    #ani.save(fn+'.gif',writer='imagemagick',fps=fps)
    #loss_ani.save('my_loss_video.gif',writer='imagemagick',fps=fps)
    #x_ani.save('my_x_video.gif',writer='imagemagick',fps=fps)

    #video = ani.to_html5_video()
    #loss_video = loss_ani.to_html5_video()
    #x_video = x_ani.to_html5_video()
    
    plt.show()
    #return video, loss_video,x_video

#plot_movie(L,W)


    

def plot_new_movie(L,W):
    fps=10
    fig,ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(14,14))
    
    ims = []

    rho=Lorenz.r
    s=Lorenz.s
    cent = np.array((0,0,18))
    r = np.linspace(0,28.8,50)
    p = np.linspace(0,2*np.pi,50)
    rr,pp = np.meshgrid(r,p)
    Z = rho-s+np.sqrt(28.8**2-rr**2)
    Z_ = rho-s-np.sqrt(28.8**2-rr*2)
    XX,YY = rr*np.cos(pp),rr*np.sin(pp)
    twenty = (rho-s)*np.ones_like(Z)

    for i in range(len(L)-400):
        ax.set_axis_off()
        im = ax.scatter(L[i,0],L[i,1],L[i,2],c='blue',s=40)
        im_ = ax.scatter(L[:i,0],L[:i,1],L[:i,2],c='blue',s=20,alpha=.5)
        imm = ax.scatter(W[i,0],W[i,1],W[i,2],c='red',s=40)
        imm_ = ax.scatter(W[:i,0],W[:i,1],W[:i,2],c='red',s=20,alpha=.5)
        
        ims.append([im,imm,im_,imm_])

    
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
    ax.scatter(L[0,0],L[0,1],L[0,2],c='blue',marker='X',s=20)
    ax.scatter(W[0,0],W[0,1],W[0,2],c='red',marker='X',s=20)
    ax.plot_surface(XX,-twenty+Z,YY+rho+1,alpha=.2,cmap=cm.cool)
    ax.plot_surface(XX,twenty-Z, YY+rho+1,alpha=.2,cmap=cm.cool)
    ax.plot(L[:,0],L[:,1],L[:,2],c='blue',lw=2,alpha=.3)
    ax.plot(W[:,0],W[:,1],W[:,2],c='red',lw=2,alpha=.3)

#    fn = 'try_hard_video'
#    ani.save(fn+'.gif',writer='imagemagick',fps=fps)
    
    plt.show()
#plot_new_movie(L,W)
#Lorenz.bounded_plot(L,W)
def okplot(L,W):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(L[:-2,0],L[:-2,1],L[:-2,2],c='blue',s=20)
    ax.scatter(W[:-2,0],W[:-2,1],W[:-2,2],c='red',s=20)
    ax.scatter(L[-2,0],L[-2,1],L[-2,2],c='blue',s=200,marker='$X$')
    ax.scatter(W[-2,0],W[-2,1],W[-2,2],c='red',s=200,marker='$U$')

    ax.plot(L[:-1,0],L[:-1,1],L[:-1,2],c='blue',alpha=.5)
    ax.plot(W[:-1,0],W[:-1,1],W[:-1,2],c='red',alpha=.5)
    ax.plot(W[-2:,0],W[-2:,1],W[-2:,2],c='orange',alpha=.9)
    
    ax.scatter(W[-1,0],W[-1,1]+.33,W[-1,2],c='orange',s=2000,marker='$G(U)$')
    ax.scatter(W[-1,0],W[-1,1],W[-1,2],c='orange',s=70)
    ax.scatter(W[-1,0]-.05,W[-1,1]-.04,W[-1,2]+.03,c='green',s=70)
    plt.show()
#okplot(L[400:408],W[400:408])

def simple_plot(L):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(L[:,0],L[:,1],L[:,2],c='black')
    plt.show()

#plot_new_movie(L,L_)
def plot_loss(L,W):
    loss = np.array([np.linalg.norm(L[i]-W[i]) for i in range(len(L))])
    plt.plot(loss)
    plt.show()

#plot_loss(L,W)

#plot_loss(L,L_)
#plot_loss(sol,sol_)
def my_rand(mean=np.array((0,.1,.2)),variance=10):
        return np.array([mean[0],mean[1]+normal(mean[1],variance),mean[2]+normal(mean[2],variance)])

def gen_data(K=K,N=N,C=C):
    Ls = []
    Ws = []
    Ys = []
    
    for k in range(K):


        r1 = normal(-10,10)
        r2 = normal(0,10)  
        r3 = normal(0,10)
        randy = np.array((r1,r2,r3))
        
        Lk,Wk = old_DA_Lorenz(randy,N=N)
        Ls.append(Lk[C:-1])
        Ws.append(Wk[C:-1])
        Ys.append(Wk[C+1:])
    L = np.array(Ls).reshape((K*(N-C-1),3))
    W = np.array(Ws).reshape((K*(N-C-1),3))
    Y = np.array(Ys).reshape((K*(N-C-1),3))
    
    return L,W,Y
tK = 400 #number of test trajectories

'''ONet'''
def DNN():
    import deepxde as dde
    L,W,Y = gen_data()
    LT,WT,YT = gen_data(K=400)

    def onet(c=0):
        datac = dde.data.Triple((W,L),Y[:,c][:,None],(WT,LT),YT[:,c][:,None])
        netc = dde.nn.DeepONet([3,100,100,100,100,100],
                               [3,100,100,100,100,100],
                               activation='relu',
                               kernel_initializer='Glorot normal',
                               use_bias=True,
                               stacked=False,
                               trainable_trunk=True)

        modelc = dde.Model(datac,netc)
        modelc.compile('adam',lr=.01)
        lhc,tsc = modelc.train(epochs=7000,display_every=1000,batch_size=40)
        
        return lhc,tsc,modelc
    
    lhx,tsx,modelx = onet(0)
    lhy,tsy,modely = onet(1)
    lhz,tsz,modelz = onet(2)


    lh = [lhx,lhy,lhz]
    ts = [tsx,tsy,tsz]
    model = [modelx,modely,modelz]
    
    return lh,ts,model

lh,ts,model = DNN()
k = len(ts[0].best_y)/NC


def get_result(NC=NC,k=k):
    Y = np.concatenate((ts[0].best_y,ts[1].best_y,ts[2].best_y),axis=1)
    X = ts[0].X_test[1]
    U = ts[0].X_test[0]
    
    Y_,X_,U_ = [],[],[]
    for i in range(int(k)):
        Y_.append(Y[i*NC:(i+1)*NC][0:-1])
        X_.append(X[i*NC:(i+1)*NC][1:])
        U_.append(U[i*NC:(i+1)*NC][1:])
    return Y_,X_,U_

Y,X,U = get_result()
Lorenz.bounded_plot(X[0],U[0],Y[0])
Lorenz.bounded_plot(X[2],U[2],Y[2])
Lorenz.bounded_plot(X[4],U[4],Y[4])
Lorenz.bounded_plot(X[7],U[7],Y[7])
Lorenz.bounded_plot(X[9],U[9],Y[9])
Lorenz.bounded_plot(X[11],U[11],Y[11])
Lorenz.bounded_plot(X[14],U[14],Y[14])
def new_norm(Y,X,U):
    l = len(Y[0])
    k = len(Y)

    YX = np.zeros(l)
    YU = np.zeros(l)
    XU = np.zeros(l)

    for i in range(k):
        yx = np.zeros(l)
        yu = np.zeros(l)
        xu = np.zeros(l)
        for n in range(l):
            yx[n] = np.linalg.norm(Y[i][n]-X[i][n])
            yu[n] = np.linalg.norm(Y[i][n]-U[i][n])
            xu[n] = np.linalg.norm(X[i][n]-U[i][n])
        YX = YX + yx
        YU = YU + yu
        XU = XU + xu
        
    return YX/k,YU/k,XU/k

yx,yu,xu = new_norm(Y,X,U)

def plot_losses(yx=yx,yu=yu,xu=xu,t_=t_):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('t')
    ax.set_ylabel('Losses')
    ax.scatter(t_[::20],yx[::20],c='magenta',label='|DNN-True|')
    ax.plot(t_,yu,c='cyan',label='|DNN-Nudge|',lw=3)
    ax.plot(t_,xu,c='black',label='|Nudge-True|')
    plt.legend()
    plt.show()

plot_losses()
##    
##        
##
##        
##            
##
##    







##
##
##import deepxde as dde
##LT,WT,YT = gen_data(K=400)
##def onet(coord=2):
##    L,W,Y = gen_data()
##    
##
##    data = dde.data.Triple((W,L),Y[:,coord][:,None],(WT,LT),YT[:,coord][:,None])
##    net = dde.nn.DeepONet([3,100,100,100,100,100],
##               [3,100,100,100,100,100],
##               activation='relu',
##               kernel_initializer='Glorot normal',
##               use_bias=True,
##               stacked=False,
##               trainable_trunk=True)
##    model = dde.Model(data,net)
##    model.compile('adam',lr=.01)
##    lh,ts = model.train(epochs=10000,display_every=1000,batch_size=50)
##    return lh,ts,model
##
##lhz,tsz,modelz = onet(2)
##lhy,tsy,modely = onet(1)
##lhx,tsx,modelx = onet(0)
##
##pz,py,px = tsz.best_y[:,0],tsy.best_y[:,0],tsx.best_y[:,0]
##
##G = np.array((px,py,pz)).T
##
##def separate(L=LT,W=WT,Y=np.array((px,py,pz)).T):
##    L_,W_,Y_ = [],[],[]
##    total_LW = np.zeros((N-C-1))
##    total_WY = np.zeros((N-C-1))
##    total_YL = np.zeros((N-C-1))
##
##    for k in range(K-2):
##        Lk = L[k*(N-C-1):(k+1)*(N-C-1)]
##        Wk = W[k*(N-C-1):(k+1)*(N-C-1)]
##        Yk = Y[k*(N-C-1):(k+1)*(N-C-1)]
##        if(k==17):
##            Lorenz.plot_vs(Lk,Wk,Yk,eta)
##        print(total_LW.shape)
##        print(Lorenz.l2_norm(Lk,Wk).shape)
##        total_LW = total_LW + Lorenz.l2_norm(Lk,Wk)
##        total_WY = total_WY + Lorenz.l2_norm(Wk,Yk)
##        total_YL = total_YL + Lorenz.l2_norm(Yk,Lk)
##    return total_LW, total_WY, total_YL
##        
##avg1,avg2,avg3 = separate()
##
##    
##
##
##        
##    
##
##def plot(L,W,Y,b=50,k=0):
##    L_=L[k]
##    W_=W[k]
##    Y_=Y[k]
##    fig = plt.figure()
##    ax = fig.add_subplot(projection='3d')
##    
##    ax.set_xlabel('x')
##    ax.set_ylabel('y')
##    ax.scatter(L_[-b:,0],L_[-b:,1],L_[-b:,2],c='blue',alpha=.5,label='True Lorenz')
##    ax.scatter(W_[-b:,0],W_[-b:,1],W_[-b:,2],c='red',alpha=.4,label='DA Lorenz',marker='D')
##    ax.scatter(Y_[-b:,0],Y_[-b:,1],Y_[-b:,2],c='green',alpha=.5,label='ONet DA Lorenz',marker='D')
##    plt.legend()
##    plt.show()
##
##
##def pl(L,Y,G,t=t,c=0):
##    #t = np.linspace(0,T,len(L))
##    fig = plt.figure(figsize=(10,8))
##    ax = fig.add_subplot()
##    
##    ax.set_xlabel('t')
##    lab = ['x','y','z']
##    ax.set_ylabel(lab[c])
##
##    ax.plot(t,L[:,c],c='blue',label='True Solution '+lab[c]+ ' (Input)',alpha=.5,lw=2)
##    plt.scatter(t[::1],Y[:,c][::1],c='green',s=60,alpha=.9,label='Nudging Solution '+lab[c]+ ' (Label)',marker='x')
##    plt.scatter(t[::1],G[:,c][::1],c='orange',s=30,alpha=1,label='DNN Solution '+lab[c]+ ' (Prediction)',marker='$O$')
##    plt.legend()
##    plt.show()
##
##
##l = LT[:num]
##y = YT[:num]
##g = G[:num]
##
##pl(l,y,g,c=0)
##pl(l,y,g,c=1)
##pl(l,y,g,c=2)
##
##
##def ploss(l,y,g,t=t):
##    ly = np.zeros_like(l[:,0])
##    lg = ly.copy()
##    yg = ly.copy()
##    for i in range(len(y)):
##        ly[i] = np.linalg.norm(l[i,:]-y[i,:])
##        lg[i] = np.linalg.norm(l[i,:]-g[i,:])
##        yg[i] = np.linalg.norm(y[i,:]-g[i,:])
##    plt.plot(t,ly,label='|True - Nudged|')
##    plt.plot(t,lg,label='|True - Prediction|')
##    plt.plot(t,yg,label='|Nudged - Prediction|')
##    plt.legend()
##    plt.show()
##
##    
##ploss(l,y,g)


##
  
