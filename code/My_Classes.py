

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import animation
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from sklearn.svm import SVR
import scipy

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import animation
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from sklearn.svm import SVR
import scipy
import matplotlib


M=35 # Number of Fourier terms to use
N=35 #
num = 100 #num = nx = nx, number of points representing x and y 
nx = num
ny = num
nt = 2000
a,b=1,1
dx = 1/(nx-1)
dy = 1/(ny-1)
dt = .002
TF = nt*dt
c=1
k=1
x = np.linspace(0,a,num,endpoint=True)
y = np.linspace(0,b,num,endpoint=True)
t = np.linspace(0,TF,nt,endpoint=True)
xx,yy = np.meshgrid(x,y)
        
z1 = np.sin(np.pi*xx/a)*np.sin(np.pi*yy/b)


class mc:
    nt = 40
    M=35 # Number of Fourier terms to use
    N=35 #
    num = 20 #num = nx = nx, number of points representing x and y 
    nx = num
    ny = num
    nt = 2000
    a,b=1,1
    dx = 1/(nx-1)
    dy = 1/(ny-1)
    dt = .01
    TF = nt*dt
    c=1
    k=1
    x = np.linspace(0,a,num,endpoint=True)
    y = np.linspace(0,b,num,endpoint=True)
    t = np.linspace(0,TF,nt,endpoint=True)
    xx,yy = np.meshgrid(x,y)
            
    z0 = np.sin(np.pi*xx/a)*np.sin(np.pi*yy/b)
    z1 = np.sin(np.pi*xx/a)*np.sin(np.pi*yy/b)
    z2 = np.sin(np.pi*xx/2)*np.sin(np.pi*yy/b)
    def __init__(self,nt=nt):
        self.nt = nt
        return None

    def st(a):
        return (a-a.mean())/(a.std())
    
    def random_surface(xx=xx,yy=yy,z1=z0,siz=19,var=.5):


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
            random_inds = np.random.choice(some_inds,size=siz,replace=False)
            
            bump_locs = some_intr[random_inds] #Hmm, why keep intr_locs then?
            bump_vals = np.zeros_like(bump_locs[:,0]) 
            
            for i in range(len(bump_vals)):
                bump_vals[i] = np.random.normal(0,var)
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
        return z1*rand_surf

    rand_surf = random_surface()
    def solution(z=rand_surf,x=x,y=y,nt=nt):

        def sin(x,m,deriv=0): #X(x)=phi, Y(y)=psi
            if(deriv==0):
                return np.sin(x*np.pi*m)
            elif(deriv==1):
                return np.pi*m*np.cos(x*np.pi*m)
            elif(deriv==2):
                return -(np.pi)**2*m**2*np.sin(x*np.pi*m)

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
            T = dt*tau*np.ones_like(px)
            if ks==True:
                return B[i,j]*px*py*ksi(T,i,j) #This represents one such surface corresp. to (m,n)=(i,j)
            else:
                return B[i,j]*px*py
        fmn = func_mn_tau(i=2,j=2)

        def deriv_x_mn_tau(i=1,j=1,tau=0):
            px,py = np.meshgrid(sin(x,i,deriv=1),sin(y,j))
            T = dt*tau*np.ones_like(px)
            return B[i,j]*px*py*ksi(T,i,j)

        def deriv_y_mn_tau(i=1,j=1,tau=0):
            px,py = np.meshgrid(sin(x,i),sin(y,j,deriv=1))
            T = tau*dt*np.ones_like(px)
            return B[i,j]*px*py*ksi(T,i,j)

        def mixed_mn_tau(i=1,j=1,tau=0):
            px,py = np.meshgrid(sin(x,i,deriv=1),sin(y,j,deriv=1))
            T = tau*dt*np.ones_like(px)
            return B[i,j]*px*py*ksi(T,i,j)
            

        def func_tau(tau=0, all_=True, M=M, N=N,ks=True,deriv=0,dvar="x"):
            if(deriv==0):
                f = np.array([[func_mn_tau(i,j,tau,ks=ks) for i in range(M)] for j in range(N)])
            elif(deriv==1):
                if(dvar=="y"):
                    f = np.array([[deriv_y_mn_tau(i,j,tau) for i in range(M)] for j in range(N)])
                elif(dvar=="x"):
                    f = np.array([[deriv_x_mn_tau(i,j,tau) for i in range(M)] for j in range(N)])
            elif(deriv==2):
                f = np.array([[mixed_mn_tau(i,j,tau) for i in range(M)] for j in range(N)])
                
            fij = np.zeros_like(f[0][0]) #sum all of the M*N functions in the Fourier series
            for i in range(N):
                for j in range(M):
                    fij+=f[i][j]
            return fij

        def sol_op(deriv=0,dvar="x"): #This returns an array of surfaces, thus subscripting the array
            surfaces = []                               # acts as though we are calling
            for i in range(nt):                         # a solution operator [from IDDS]
                surface = func_tau(tau=i,deriv=deriv,dvar=dvar)                   # at time i/nt
                surfaces.append(surface)
            return surfaces
        u = np.array(sol_op())
        #ux = np.array(sol_op(deriv=1))
        #uy = np.array(sol_op(deriv=1,dvar="y"))
        #uxy = np.array(sol_op(deriv=2))
        return u,fmn


    def FDM(un=rand_surf,nt=nt):
        u = un
        v = u.copy()
        vn = v.copy()
        un = u.copy()
        U = np.zeros((nt,ny,nx))
        
        for n in range(nt):
            un_1 = un.copy()
            un = u.copy()

            u[1:-1, 1:-1] = dt**2*(1/dx**2 *(un[0:-2, 1:-1]-2*un[1:-1,1:-1]+un[2:, 1:-1] )+1/dy**2 *(un[1:-1, 0:-2]-2*un[1:-1, 1:-1] + un[1:-1, 2:]))+2*un[1:-1, 1:-1] - un_1[1:-1, 1:-1]

            u[:,0]=0
            u[:,-1]=0
            u[0,:]=0
            u[-1,:]=0

            U[n]=u

        return U

    #fmn = solution(nt=1)[1] #this is a sin(x) X sin(y) with 2 valleys and 2 mountains

    def FDM_V(U_,x=x,y=y,xx=xx,yy=yy,nt=nt):
        #vn=fmn
        vn = np.zeros_like(U_[0])
        nt = len(U_)
        
        def interp(z,x,y):
            f = scipy.interpolate.interp2d(x,y,z,kind='linear')
            F = np.zeros((nx,ny))

            for i in range(1,nx-1):
                for j in range(1,ny-1):
                    F[i,j] = f(xx[i,j],yy[i,j])
            return F

        nu = 3
        v = vn.copy()

        partial_v = v[::2,::2]
        partial_v = np.concatenate((partial_v,np.zeros((1,len(partial_v)))),axis=0)
        partial_v = np.concatenate((partial_v,np.zeros((len(partial_v),1))),axis=1)
        partial_vn = partial_v.copy()

        partial_x = x[::2]
        partial_y = y[::2]
        partial_x = np.append(partial_x,1)
        partial_y = np.append(partial_y,1)


        shp = vn.shape
        V = np.zeros((nt,shp[1],shp[0]))
        for n in range(nt):
            vn_1 = vn.copy()
            vn = v.copy()
            partial_vn = vn[::2,::2]
            partial_vn = np.concatenate((partial_vn,np.zeros((1,len(partial_vn)))),axis=0)
            partial_vn = np.concatenate((partial_vn,np.zeros((len(partial_vn),1))),axis=1)

            partial_un = U_[n][::2,::2]
            partial_un = np.concatenate((partial_un,np.zeros((1,len(partial_un)))),axis=0)
            partial_un = np.concatenate((partial_un,np.zeros((len(partial_un),1))),axis=1)

            nu_term = partial_un.copy()
            nu_term[1:-1,1:-1] = dt**2*nu*(partial_vn[1:-1,1:-1]-partial_un[1:-1,1:-1])

            Nu_Term = interp(nu_term,partial_x,partial_y)

            v[1:-1,1:-1] = dt**2*(1/dx**2 *(vn[0:-2, 1:-1]-2*vn[1:-1,1:-1]+vn[2:, 1:-1] )+1/dy**2 *(vn[1:-1, 0:-2]-2*vn[1:-1, 1:-1] + vn[1:-1, 2:]))+2*vn[1:-1, 1:-1] - vn_1[1:-1, 1:-1]\
                           -Nu_Term[1:-1,1:-1]
                           
                                        

            v[:,0]=0
            v[:,-1]=0
            v[0,:]=0
            v[-1,:]=0

            V[n] = v
        return V


    def p2(a,b,scatter=False):
        fig = plt.figure()
        x = np.linspace(0,1,a.shape[1])
        y = np.linspace(0,1,a.shape[0])
        x,y = np.meshgrid(x,y)

        ax= fig.add_subplot(projection='3d')
        if(scatter==True):
            ax.scatter(x,y,a)
            ax.scatter(x,y,b)
        else:
            ax.scatter(x[::3,::3],y[::3,::3],a[::3,::3],alpha=.5)
            ax.plot_surface(x,y,b,alpha=.5)
        plt.show()


    def pipeplots(a,b):
        fig = plt.figure()


    def dot_plot(a):
        fig = plt.figure()
        x = np.linspace(0,1,a[0].shape[0])
        y = np.linspace(0,1,a[0].shape[1])
        x,y = np.meshgrid(x,y)
        X,Y = x[::10,::10],y[::10,::10]

        ax= fig.add_subplot(projection='3d')
        ax.set_axis_off()
        ax.axes.set_zlim3d(bottom=-.47,top=.47)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.scatter(x[37,29],y[37,29],a[21][37,29],c='red',alpha=1,marker='D',s=44)
        ax.scatter(x[37,29],y[37,29],a[19][37,29],c='red',alpha=.7,marker='D',s=44)
        ax.scatter(x[37,29],y[37,29],a[17][37,29],c='red',alpha=.6,marker='D',s=44)
        ax.scatter(x[37,29],y[37,29],a[15][37,29],c='red',alpha=.5,marker='D',s=44)
        ax.scatter(x[37,29],y[37,29],a[13][37,29],c='red',alpha=.4,marker='D',s=44)
        ax.scatter(x[37,29],y[37,29],a[11][37,29],c='red',alpha=.3,marker='D',s=44)
        ax.scatter(x[37,29],y[37,29],a[9][37,29],c='red',alpha=.2,marker='D',s=44)
        ax.scatter(x[37,29],y[37,29],a[7][37,29],c='red',alpha=.1,marker='D',s=44)
        
        ax.plot_surface(x,y,a[21],alpha=.5,cmap=cm.viridis)
        
        ax.scatter(X,Y,a[19][::10,::10],alpha=.6,c='black',s=1)
        ax.scatter(X,Y,a[17][::10,::10],alpha=.5,c='black',s=.8)
        ax.scatter(X,Y,a[15][::10,::10],alpha=.4,c='black',s=.8)
        ax.scatter(X,Y,a[13][::10,::10],c='black',alpha=.4,s=.6)
        ax.scatter(X,Y,a[11][::10,::10],c='black',alpha=.3,s=.6)
        ax.scatter(X,Y,a[9][::10,::10],c='black',alpha=.2,s=1.4)
        ax.scatter(X,Y,a[7][::10,::10],c='black',alpha=.1,s=1.4)

        ax.scatter(x[::10,-1],y[::10,-1],np.zeros_like(x[::10,-1]),c='black',alpha=.5,s=1)
        ax.scatter(x[::10,0],y[::10,0],np.zeros_like(x[::10,0]),c='black',alpha=.5,s=1)
        ax.scatter(x[0,::10],y[0,::10],np.zeros_like(x[0,::10]),c='black',alpha=.5,s=1)
        ax.scatter(x[-1,::10],y[-1,::10],np.zeros_like(x[-1,::10]),c='black',alpha=.5,s=1)

        ax.scatter(x[37,29],y[37,29],a[21][37,29],c='red',alpha=1,marker='D')
        ax.scatter(x[37,29],y[37,29],a[19][37,29],c='red',alpha=.7,marker='D')
        ax.scatter(x[37,29],y[37,29],a[17][37,29],c='red',alpha=.6,marker='D')
        ax.scatter(x[37,29],y[37,29],a[15][37,29],c='red',alpha=.5,marker='D')
        ax.scatter(x[37,29],y[37,29],a[13][37,29],c='red',alpha=.4,marker='D')
        ax.scatter(x[37,29],y[37,29],a[11][37,29],c='red',alpha=.3,marker='D')
        ax.scatter(x[37,29],y[37,29],a[9][37,29],c='red',alpha=.2,marker='D')
        ax.scatter(x[37,29],y[37,29],a[7][37,29],c='red',alpha=.1,marker='D')


        plt.show()



    def new_FDM_V(U_,x=x,y=y,xx=xx,yy=yy):
        #vn=fmn
        vn = np.zeros_like(U_[0])
        nt = len(U_)
        
        def interp(z,x,y):
            f = scipy.interpolate.interp2d(x,y,z,kind='linear')
            F = np.zeros((nx,ny))

            for i in range(1,nx-1):
                for j in range(1,ny-1):
                    F[i,j] = f(xx[i,j],yy[i,j])
            return F

        nu = 7
        v = vn.copy()

        partial_v = v[::2,::2]
        partial_v = np.concatenate((partial_v,np.zeros((1,len(partial_v)))),axis=0)
        partial_v = np.concatenate((partial_v,np.zeros((len(partial_v),1))),axis=1)
        partial_vn = partial_v.copy()

        partial_x = x[::2]
        partial_y = y[::2]
        partial_x = np.append(partial_x,1)
        partial_y = np.append(partial_y,1)


        shp = vn.shape
        V = np.zeros((nt,shp[1],shp[0]))
        for n in range(nt):
            vn_1 = vn.copy()
            vn = v.copy()
            partial_vn = vn[::2,::2]
            partial_vn = np.concatenate((partial_vn,np.zeros((1,len(partial_vn)))),axis=0)
            partial_vn = np.concatenate((partial_vn,np.zeros((len(partial_vn),1))),axis=1)

            partial_un = U_[n][::2,::2]
            partial_un = np.concatenate((partial_un,np.zeros((1,len(partial_un)))),axis=0)
            partial_un = np.concatenate((partial_un,np.zeros((len(partial_un),1))),axis=1)
            
        

            nu_term = partial_un.copy()
            nu_term[1:-1,1:-1] = partial_un[1:-1,1:-1]

            Nu_Term = nu*interp(nu_term,partial_x,partial_y)
            
            #if(n%20==0):
             #   mc.p2(partial_vn,partial_un)
              #  mc.p2(Nu_Term,Nu_Term)

            v[1:-1,1:-1] = 1/(1+nu*dt**2)*\
                           (dt**2*(1/dx**2 *(vn[0:-2, 1:-1]-2*vn[1:-1,1:-1]+vn[2:, 1:-1] )+1/dy**2 *(vn[1:-1, 0:-2]-2*vn[1:-1, 1:-1] + vn[1:-1, 2:])-Nu_Term[1:-1,1:-1])\
                            +2*vn[1:-1, 1:-1] - vn_1[1:-1, 1:-1])
                           
                           
                                        

            v[:,0]=0
            v[:,-1]=0
            v[0,:]=0
            v[-1,:]=0

            V[n] = v
        return V










    


    def norm(u,v):
        d = np.zeros_like(u[:,0,0])
        for i in range(len(d)):
            d[i] = np.linalg.norm(u[i,:,:]-v[i,:,:])
        return d

    def pl(A,B,C,x,y,color='red'):
        xx,yy = np.meshgrid(x,y)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_axis_off()
        ax.set_zlim(top=.4,bottom=-.4)
        ax.plot_surface(xx,yy,A,cmap=cm.rainbow)
        #ax.scatter(xx,yy,A,c='black')
        ax.plot_surface(xx,yy,B,cmap=cm.Reds,alpha=.3)
        ax.plot_surface(xx,yy,C,cmap=cm.Blues_r,alpha=.3)
        plt.show()

    def ppp(A,B,C,x,y):
        xx,yy = np.meshgrid(x,y)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_axis_off()
        ax.set_zlim(top=.4,bottom=-.4)
        #ax2 = fig.add_subplot(projection='3d')
        #ax2.set_axis_off()
        #ax2.set_zlim(top=.4,bottom=.4)

        ax.plot_surface(xx,yy,B)
        ax.plot_surface(xx,yy,C)
        #ax2.plot_surface(xx,yy,C)
        plt.show()
        

    def pll(A,B,C,x,y):
        xx,yy = np.meshgrid(x,y)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_axis_off()
        ax.set_zlim(top=.4,bottom=-.4)
        
        ax.plot_surface(xx[10:20,10:20],yy[10:20,10:20],A[10:20,10:20],cmap=cm.rainbow,alpha=.5)
        #ax.scatter(xx[10:20,10:20],yy[10:20,10:20],A[10:20,10:20],c='black')
        #ax.scatter(xx[10:20,10:20],yy[10:20,10:20],B[10:20,10:20],c='red',alpha=.5)
        #ax.scatter(xx[10:20,10:20],yy[10:20,10:20],C[10:20,10:20],c='blue',alpha=.5)

        ax.scatter(xx[13,13],yy[13,13],A[13,13],alpha=1,c='black',s=100)
        ax.scatter(xx[13,13],yy[13,13],B[13,13],alpha=1,c='red',s=50)
        ax.scatter(xx[13,13],yy[13,13],C[13,13],alpha=1,c='blue',s=50)

        ax.plot(np.array((xx[13,13],xx[13,13])),np.array((yy[13,13],yy[13,13])),np.linspace(A[13,13],B[13,13],2),c='red')
        ax.plot(np.array((xx[13,13],xx[13,13])),np.array((yy[13,13],yy[13,13])),np.linspace(A[13,13],C[13,13],2),c='blue')
        
        ax.plot_surface(xx[10:20,10:20],yy[10:20,10:20],B[10:20,10:20],cmap=cm.Reds,alpha=.7)
        ax.plot_surface(xx[10:20,10:20],yy[10:20,10:20],C[10:20,10:20],cmap=cm.Blues_r,alpha=.7)
        plt.show()

    def p(y):
        x = np.linspace(0,TF,len(y))
        plt.scatter(x,y)
        plt.show()

    def p2(a,b,scatter=False):
        fig = plt.figure()
        x = np.linspace(0,1,a.shape[0])
        y = np.linspace(0,1,a.shape[1])
        x,y = np.meshgrid(x,y)

        ax= fig.add_subplot(projection='3d')
        if(scatter==True):
            ax.scatter(x,y,a)
            ax.scatter(x,y,b)
        else:
            ax.scatter(x[::3,::3],y[::3,::3],a[::3,::3],alpha=.5,c='green')
            ax.plot_surface(x,y,b,alpha=.5,cmap=cm.viridis)
        plt.show()

    def plot_movie(U,scale=1,bottom=-.7,top=.7):
        x = np.linspace(xx[0,0],xx[0,-1],len(U[0]))
        y = np.linspace(yy[0,0],yy[-1,0],len(U[0,0]))
        fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
        X,Y = np.meshgrid(x,y)
        ims = []


        for i in range(len(U)):
            ax.axes.set_zlim3d(bottom=bottom,top=top)
            ui = U[int(i)]
            im = ax.plot_surface(X[::scale,::scale],Y[::scale,::scale],ui[::scale,::scale],animated=True,cmap=cm.rainbow,alpha=1)
            
            ims.append([im])
        ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
        plt.show()
        return ani

    def plot_heat_movie(U,scale=1):
        x = np.linspace(xx[0,0],xx[0,-1],len(U[0]))
        y = np.linspace(yy[0,0],yy[-1,0],len(U[0,0]))
        fig,ax = plt.subplots()
        X,Y = np.meshgrid(x,y)
        ims = []

        for i in range(len(U)):
            ui = U[int(i)]
            im = ax.contourf(X[::scale,::scale],Y[::scale,::scale],ui[::scale,::scale],animated=True,cmap=cm.rainbow,alpha=1)
            
            ims.append([im])
        ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
        #plt.colorbar()
        plt.show()
        return ani
        
    

    def plot_movies(U,V,scale=1,bottom=-.3,top=.3,save=False):
        x = np.linspace(xx[0,0],xx[0,-1],len(U[0]))
        y = np.linspace(yy[0,0],yy[-1,0],len(U[0,0]))
        fig,ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(11,11))
        loss_fig, loss_ax = plt.subplots(figsize=(7,5))
        X,Y = np.meshgrid(x,y)
        preim = []
        ims = []
        imms = []
        loss_ims = []
        
        losses = np.array([np.linalg.norm(U[i]-V[i]) for i in range(len(U))])
        T_ = np.linspace(0,2,len(U))

        for i in range(len(U)):
            ax.axes.set_zlim3d(bottom=bottom,top=top)
            ax.set_axis_off()
            ui = U[int(i)]
            vi = V[int(i)]
            I = ax.plot_surface(X[::scale,::scale],Y[::scale,::scale],ui[::scale,::scale],animated=True,cmap=cm.rainbow,alpha=.7)
            #ii = ax.plot_surface(X[::scale,::scale],Y[::scale,::scale],vi[::scale,::scale],animated=True,cmap=cm.magma,alpha=.3)
            im = ax.plot_wireframe(X[::scale,::scale],Y[::scale,::scale],ui[::scale,::scale],animated=True,color='blue',alpha=.7,)
            imm = ax.scatter(X[::scale,::scale],Y[::scale,::scale],vi[::scale,::scale],animated=True,alpha=.9,c='black',s=17)
            loss_im = loss_ax.scatter(T_[:i],losses[:i],c='black',s=8)
            ims.append([I,im,imm])
            loss_ims.append([loss_im])
        ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
        loss_ani = animation.ArtistAnimation(loss_fig,loss_ims,interval=50,blit=True)
        if(save==True):
            ani.save('fallbackintoplace3.gif',writer='ok',fps=10)
            loss_ani.save('youwideeyedgirls3.gif',writer='ok',fps=10)
        plt.show()
        return ani,loss_ani

    def plot_vector_movie(U,V,x,y,fig_size=(10,5),sc=3):
        ny = len(U[0])
        nx = len(U[0,0])
        SU=np.zeros((ny,nx))
        SV=np.zeros((ny,nx))

        for i in range(len(U)):
          SU+=U[i,:,:]
          SV+=V[i,:,:]

        #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        fig, ax = plt.subplots(figsize=fig_size,dpi=50)
        X, Y = np.meshgrid(x, y)

        preim = []
        ims = []
        for i in range(len(U)):
            ui = U[i]
            vi = V[i]
            #pi = P[i]
            #im = ax.imshow(ui,animated=True)
            #I = ax.contourf(X,Y,P[i])
            im = ax.quiver(X[::sc,::sc],Y[::sc,::sc],ui[::sc,::sc],vi[::sc,::sc],animated=True,color='deepskyblue',lw=.0001,scale=11)
            imm = ax.scatter(ui,Y,c='black')
            #ps = ax.contourf(X[::3,::3],Y[::3,::3],P[i][::3,::3],alpha=.01,animated=True,cmap=cm.viridis)
            ims.append([im,imm])

        ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
        plt.show()


    def plot_vector_movies(U,V,A,B,x,y,fig_size=(10,5),sc=3):
        ny = len(U[0])
        nx = len(U[0,0])
        SU=np.zeros((ny,nx))
        SV=np.zeros((ny,nx))

        for i in range(len(U)):
          SU+=U[i,:,:]
          SV+=V[i,:,:]

        #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        fig, ax = plt.subplots(figsize=fig_size,dpi=50)
        X, Y = np.meshgrid(x, y)

        preim = []
        ims = []
        for i in range(len(U)):
            ui = U[i]
            vi = V[i]
            ai = A[i]
            bi = B[i]
            #pi = P[i]
            #im = ax.imshow(ui,animated=True)
            #I = ax.contourf(X,Y,P[i])
            im = ax.quiver(X[::sc,::sc],Y[::sc,::sc],ui[::sc,::sc],vi[::sc,::sc],animated=True,color='deepskyblue',lw=.0001,scale=11)
            imm = ax.scatter(ui,Y,c='black')
            IMM = ax.scatter(ai,Y,c='red')
            #ps = ax.contourf(X[::3,::3],Y[::3,::3],P[i][::3,::3],alpha=.01,animated=True,cmap=cm.viridis)
            ims.append([im,imm,IMM])

        ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
        plt.show()
        
    def plot_new_vector_movie(U,V,x,y,fig_size=(10,5),sc=3):
        ny = len(U[0])
        nx = len(U[0,0])
        SU=np.zeros((ny,nx))
        SV=np.zeros((ny,nx))

        for i in range(len(U)):
          SU+=U[i,:,:]
          SV+=V[i,:,:]

        #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        fig, ax = plt.subplots(figsize=fig_size,dpi=50)
        X, Y = np.meshgrid(x, y)

        preim = []
        ims = []
        for i in range(len(U)):
            ui = U[i]
            vi = V[i]
            #pi = P[i]
            #im = ax.imshow(ui,animated=True)
            #I = ax.contourf(X,Y,P[i])
            im = ax.quiver(X[::sc,::sc],Y[::sc,::sc],ui[::sc,::sc],vi[::sc,::sc],animated=True,color='deepskyblue',lw=.0001,scale=11)
            #ps = ax.contourf(X[::3,::3],Y[::3,::3],P[i][::3,::3],alpha=.01,animated=True,cmap=cm.viridis)
            ims.append([im])

        ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
        plt.show()

from numpy.random import uniform
from numpy.random import normal
from numpy.random import randint

class Lorenz:
    N = 1000
    M = 20
    NI = 100
    lr = .002

    j=20 #This is like 1/h from BMFT
    nu=30 #This is like beta in BMFT

    t0 = N-NI
    tf = N

    eta = .01
    T = eta*N

    t = np.linspace(0,T,N,endpoint=False)
    t_ = np.linspace(eta,T+eta,N,endpoint=False)

    I = t[t0:tf]
    I_ = t_[t0:tf]

    L0 = np.array((0,0.1,0.2))
    W0 = np.array((0, normal(.1,1), normal(.2,1)))

    s = 10
    r = 28
    b = 8/3
    
    def step(L,W):
        s = Lorenz.s
        r = Lorenz.r
        b = Lorenz.b
        nu = Lorenz.nu
        eta = Lorenz.eta
        
        dx = s*(L[1]-L[0])
        dy = r*L[0]-L[1]-L[0]*L[2]
        dz = L[0]*L[1]-b*L[2]
        dL = np.array((dx,dy,dz))

        Dx = s*(W[1]-W[0])-nu*(W[0]-L[0])
        Dy = r*W[0]-W[1]-W[0]*W[2]
        Dz = W[0]*W[1]-b*W[2]
        DW = np.array((Dx,Dy,Dz))
        
        L_ = L+eta*dL
        W_ = W+eta*DW
        return L_,W_
    
    def l2_norm(x1,x2):
        l2 = np.sqrt((x1[:,0] - x2[:,0])**2 + (x1[:,1] - x2[:,1])**2 + (x1[:,2] - x2[:,2])**2)
        return l2
    
    def DA_Lorenz(L_0=L0, W_0=W0, n=N , nu=nu): #n is number of timestamps, L_0, W_0 are ICs

        lorenz = np.zeros((n,3))
        worenz = lorenz.copy()
        
        lorenz[0]=L_0
        worenz[0]=W_0
        
        for i in range(1,n):
            lorenz[i],worenz[i] = Lorenz.step(lorenz[i-1],worenz[i-1])

        return lorenz, worenz

    def norm(L,W,t):
        dists = np.zeros((len(L)))
        for i in range(len(dists)):
            dists[i] = np.linalg.norm(L[i]-W[i])
        plt.plot(t,dists,c='black')
        plt.show()
        return dists


    def make_data(lorenz,worenz,n=10):
        new_l = np.zeros((n,3))
        new_w = np.zeros((n,3))

        L = np.concatenate((lorenz,new_l))
        W = np.concatenate((worenz,new_w))

        data = []
        for i in range(len(lorenz),len(L)):
            L[i],W[i] = Lorenz.step(L[i-1],W[i-1])
            datum = [W[i-1],L[i-1],W[i]]
            data.append(datum)
        return data

    def pl(L,eta):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(L[-100:,0],L[-100:,1],L[-100:,2],c='red')
        ax.scatter(L[:,0],L[:,1],L[:,2],c=np.arange(len(L)),s=20,alpha=.8)
        ax.scatter(L[0,0],L[0,1],L[0,2],c='black',s=40,Label='Initial Position '+str(L[0]))
        ax.scatter(L[-100:,0],L[-100:,1],L[-100:,2],c='red',label='Last 100, with Final Position '+str(L[-1]))
        
        cbar = plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,len(L)*eta), cmap=cm.viridis),ax=ax)
        cbar.set_label('time')
        plt.legend()
        plt.show()

    def pl2(L,W):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(L[:,0],L[:,1],L[:,2],c='blue')
        ax.scatter(W[:,0],W[:,1],W[:,2],c='red')
        plt.show()

    def plot_full_lorenz(lorenz,worenz):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.set_axis_off()
        ax.set_title(label='||L0-W0|| = '+str(np.linalg.norm(lorenz[0]-worenz[0])))
        ax.scatter(lorenz[:,0],lorenz[:,1],lorenz[:,2],c='blue',alpha=.5,s=2)
        ax.plot(lorenz[:,0],lorenz[:,1],lorenz[:,2],c='blue',alpha=.5)
        ax.scatter(worenz[:,0],worenz[:,1],worenz[:,2],c='red')
        ax = fig.add_subplot(1,1,2,projection='3d')
        ax.scatter(lorenz[-200:,0],lorenz[-200:,1],lorenz[-200:,2],c='blue')
        ax.scatter(worenz[-200:,0],worenz[-200:,1],worenz[-200:,2],c='red')
        plt.show()

    def plot_one_coord(lorenz,worenz,coord=0):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(Lorenz.t,lorenz[:,coord])
        ax.plot(Lorenz.t,worenz[:,coord])
        plt.show()

    def plot_norm(lorenz,worenz,eta,lab_n=0):
        
        Label = ['| u(t)-x(t)|','| Gu(t)-u(t)|','| Gu(t)-x(t)|']
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('t')
        ax.set_ylabel('Error')
        plt.plot(np.linspace(0,len(lorenz)*eta,len(lorenz)),Lorenz.l2_norm(lorenz,worenz),label=Label[lab_n],c='black')
        plt.legend()
        #plt.show()

    def sanity(data):
        sanity_check = Lorenz.step(data[-1][1],data[-1][0])[1]==data[-1][2]
        print(sanity_check)

    def plot_vs(L,W,G,eta):
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(L[:,0],L[:,1],L[:,2],c='blue',label='DA Lorenz',alpha=.4)
        ax.scatter(W[:,0],W[:,1],W[:,2],c='red',label='Test DA Lorenz',alpha=.4)
        ax.scatter(G[:,0],G[:,1],G[:,2],c='green',label='ONet Predication DA Lorenz',alpha=.4)
        plt.legend()
        Lorenz.plot_norm(L,W,eta,lab_n=0)
        Lorenz.plot_norm(W,G,eta,lab_n=1)
        Lorenz.plot_norm(L,G,eta,lab_n=2)
        plt.show()


    
    def bounded_plot(lorenz,worenz,donenz):
        rho=Lorenz.r
        s=Lorenz.s
        cent = np.array((-9,-7,25))
        r = np.linspace(0,31,50)
        p = np.linspace(0,2*np.pi,50)
        rr,pp = np.meshgrid(r,p)
        Z = rho-s+np.sqrt(31**2-rr**2)
        Z_ = rho-s-np.sqrt(31**2-rr*2)
        XX,YY = rr*np.cos(pp),rr*np.sin(pp)
        twenty = (rho-s)*np.ones_like(Z)

        center = np.array((0,0,rho-s))

        fig = plt.figure(figsize=(11,11))
        ax = fig.add_subplot(projection='3d')
        ax.set_axis_off()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.plot_surface(XX,-twenty+Z,YY+rho+1,alpha=.14,cmap=cm.winter)
        ax.plot_surface(XX,twenty-Z, YY+rho+1,alpha=.14,cmap=cm.winter)
        ax.plot(lorenz[:5000,0][::1],lorenz[:5000,1][::1],lorenz[:5000,2][::1],c='blue',alpha=.4,lw=1.5,label='True Lorenz')
        ax.scatter(lorenz[:5000,0][::1],lorenz[:5000,1][::1],lorenz[:5000,2][::1],c='blue',alpha=.4,s=10,label='True Lorenz')

        ax.scatter(donenz[:5000,0][::1],donenz[:5000,1][::1],donenz[:5000,2][::1],c='orange',alpha=.7,marker='$X$',s=75,label='DeepONet DA Lorenz')
        ax.plot(worenz[:5000,0],worenz[:5000,1],worenz[:5000,2],c='green',alpha=.9,lw=3,label='DA Lorenz')
        ax.plot(np.linspace(0,31,100),np.zeros((100)),(rho+1)*np.ones((100))+np.zeros((100)),c='black')
        ax.plot(np.zeros((100)),np.linspace(0,31,100),(rho+1)*np.ones((100))+np.zeros((100)),c='black')
        ax.plot(np.zeros((100)),np.zeros((100)),(rho+1)*np.ones((100))+np.linspace(0,31,100),c='black')
        #lol = np.linspace(0,18,50)
        #ax.plot(lol,lol,(rho+1) + lol,c='red')


        ax.scatter(33,0,rho+1,c='black',marker='$x$')
        ax.scatter(0,33,rho+1,c='black',marker='$y$')
        ax.scatter(0,0,33+rho+1,c='black',marker='$z$')
        plt.legend()
        plt.show()


        #plt.legend()


        








    




