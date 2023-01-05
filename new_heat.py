import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import scipy
#from My_Classes import mc

K=50 # Number of trajectories
num = 20 #num = nx = nx, number of points representing x and y 
nx = num
ny = num
nt = 300

start_nt = 0
end_nt = nt
indices_nt = nt-start_nt

a,b=1,1
dx = 1/(nx-1)
dy = 1/(ny-1)
dt = .001
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

def interp(z,x=x,y=y):
    return scipy.interpolate.interp2d(x,y,z,kind='cubic',fill_value=4)

i_x = np.linspace(x[0],x[-1],100,endpoint=True)
i_y = i_x.copy()
        

#W = nudge(rand_surf)
#W,i_W = i_nudge(rand_surf)

def i_nudge(wn,U=U):
    w = wn
    W = np.zeros((nt,ny,nx))
        
    interp_w = interp(w)
    i_w = interp_w(i_x,i_y)

    i_W = np.zeros((nt,100,100))

    for n in range(nt):
        wn = w.copy()
        
        w[1:-1,1:-1] = dt*(\
            alpha*(1/dx**2 *(wn[0:-2, 1:-1]-2*wn[1:-1,1:-1]+wn[2:, 1:-1] )+1/dy**2 *(wn[1:-1, 0:-2]-2*wn[1:-1, 1:-1] + wn[1:-1, 2:]))\
            - mu*(wn[1:-1,1:-1]-U[n][1:-1,1:-1])) + wn[1:-1,1:-1]
        
        w[:,0]=0
        w[:,-1]=0
        w[0,:]=0
        w[-1,:]=0

        interp_wn = interp(w)
        i_wn = interp_wn(i_x,i_y)

        W[n]=w
        i_W[n]=i_wn
    return W,i_W
        

#W = nudge(rand_surf)
W,i_W = i_nudge(rand_surf)

def plot_movies(U,V,lab='temp'):
    fig,ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(11,11))
    X,Y = np.meshgrid(x,y)
    preim = []
    ims = []

    ix = np.linspace(x[0],x[-1],len(V[0]),endpoint=True)
    iy = ix.copy()
    IX,IY = np.meshgrid(ix,iy)

    for i in range(len(U)):
        ax.axes.set_zlim3d(bottom=-.3,top=.3)
        ax.set_axis_off()
        ui = U[int(i)]
        vi = V[int(i)]

        im = ax.plot_surface(IX,IY,vi,animated=True,cmap=plt.get_cmap('rainbow'),alpha=.5)
        imm = ax.scatter(X,Y,ui,animated=True,alpha=.9,c='black',s=10)
        ims.append([im,imm])
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)

    #ani.save('a.gif',writer='imagemagick',fps=10)
    plt.show()

plot_movies(U,i_W[:,::1,::1])
