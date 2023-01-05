from mpl_toolkits.mplot3d import Axes3D 

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import numpy as np

##import deepxde as dde
##from deepxde.data import Triple as Triple
##from deepxde.nn import DeepONet as ONet
from matplotlib import cm
  
# importig movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

nx = 200
ny = 70
nt = 200
c = 3
dx = 3 / (nx - 1)
dy = 1 / (ny - 1)
dt = .01

y1=0
y2=1

rho = 1 
nu = .01
F = 3
time = np.arange(0,nt*dt,dt)

#initial conditions
u = np.zeros((ny, nx))
un = np.zeros((ny, nx))

v = np.zeros((ny, nx))
vn = np.zeros((ny, nx))

p = np.ones((ny, nx))
pn = np.ones((ny, nx))

#Domain
dom = np.load(r'D:\....2022\Math 881\Mid_Semester\pipe_dom.npz')
X = dom['x_dom']
Y = dom['y_dom']

def load_data(): #See "pipeflow" code for creation of data
    U = np.load(r'D:\....2022\Math 881\Mid_Semester\u_comp.npz')['u_comp']
    V = np.load(r'D:\....2022\Math 881\Mid_Semester\v_comp.npz')['v_comp']
    P = np.load(r'D:\....2022\Math 881\Mid_Semester\pressure.npz')['pressure']
    return U,V,P
##
##def domain():
##    X_ = np.ravel(X)[:,None]
##    Y_ = np.ravel(Y)[:,None]
##    T_ = np.ones_like(X_)
##    D_ = np.concatenate((X_,Y_),axis=1)
##
##    SD_ = []
##    ST_ = []
##    for i in range(nt):
##        SD_.append(D_)
##        ST_.append(time[i]*T_)
##    SD_ = np.array(SD_)
##    SD_ = SD_.reshape((SD_.shape[0]*SD_.shape[1] , SD_.shape[2]))
##    ST_ = np.array(ST_)
##    ST_ = ST_.reshape((ST_.shape[0]*ST_.shape[1] , ST_.shape[2]))
##
##    Dom = np.concatenate((SD_, ST_),axis=1)
##    return Dom
##
##def stack_ic(ic):
##    ic_stack = []
##    for i in range(nt):
##        ic_stack.append(ic.ravel()[:,None])
##    return np.array(ic_stack).reshape((nt*nx*ny,1))
##
##
##def arrange_data():
##    U,V,P = load_data()
##    dom = domain()
##    su = stack_ic(u)
##    sv = stack_ic(v)
##    sp = stack_ic(p)
##    U = U.reshape((U.size,1))
##    V = V.reshape((V.size,1))
##    P = P.reshape((P.size,1))
##    return U,V,P,dom,su,sv,sp
##
##def odata():
##    SU,SV,SP,SD,su,sv,sp = arrange_data()
##
##    data_u = Triple((su,SD),SU,(su,SD),SU)
##    data_v = Triple((sv,SD),SV,(sv,SD),sv)
##    data_p = Triple((sp,SD),SP,(sp,SD),sp)
##    return data_u, data_v, data_p
##
##data_u, data_v, data_p = odata()
##
##def onet(data,epochs=8000):
##    net = ONet([1,140,200,200,140],[3,140,200,200,140],
##               activation='relu',kernel_initializer='Glorot normal',
##               use_bias=True,
##               stacked=False,
##               trainable_trunk=True)
##
##    model = dde.Model(data,net)
##    model.compile('adam',lr=.01)
##    lh, ts = model.train(epochs=epochs,batch_size=80,display_every=int(epochs/4))
##    return lh, ts, model
##
##lhu, tsu, modelu = onet(data_u,epochs=28000)
##lhv, tsv, modelv = onet(data_v,epochs=200)
##lhp, tsp, modelp = onet(data_p,epochs=1000)
##
##pred_u = tsu.best_y.reshape((nt,ny,nx))
##pred_v = tsv.best_y.reshape((nt,ny,nx))
##pred_p = tsp.best_y.reshape((nt,ny,nx))
##
##def plot_movie(U,V):
##    U_,V_,P_ = load_data()
##    cU=np.zeros((ny,nx))
##    cV=np.zeros((ny,nx))
##
##    for i in range(len(U)):
##      cU+=U[i,:,:]
##      cV+=V[i,:,:]
##
##    #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
##    fig, ax = plt.subplots(figsize=(30,14),dpi=50)
##
##    preim = []
##    ims = []
##
##    
##    for i in range(len(U)):
##        ui = U[i]
##        vi = V[i]
##        #pi = P[i]
##        #im = ax.imshow(ui,animated=True)
##        im = ax.quiver(X[::5,::5],Y[::5,::5],ui[::5,::5],vi[::5,::5],animated=True,color='black',lw=.0001,scale=11)
##        imm = ax.scatter(ui,Y,c='red',s=180)
##        immm = ax.scatter(U_[i],Y,c='white',marker='$+$',s=400,alpha=1)
##        #ps = ax.quiver(Y,X,vi,ui,animated=True,color='orange',lw=.0001)
##        #if i==len(U)-1:
##         #   ims.append([im,imm,immm,ok])
##        #elif(i<len(U)-1):
##        #    ims.append([im,imm,immm])
##        ims.append([im,imm,immm])
##    ax.contourf(X,Y,U[-1],levels=np.linspace(0,10,80),cmap=cm.Blues_r,animated=False,alpha=.6)
##    ani = animation.ArtistAnimation(fig,ims,interval=10,blit=True,)
##    #ani.save('nseonet.gif',writer='ok',fps=10)
##    plt.show()
##

def plot_still():
    U_,V_,P_ = load_data()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(U_[-1],Y,c='black',lw=5)
    ax.contourf(X,Y,U_[-1],levels=np.linspace(0,15,200),cmap=cm.rainbow,animated=False,alpha=1)
    #ax.quiver(X[::3,::3],Y[::3,::3],U_[-1][::3,::3],V_[-1][::3,::3],color='black',lw=.0001,scale=10)
    plt.show()

#plot_movie()
plot_still()
