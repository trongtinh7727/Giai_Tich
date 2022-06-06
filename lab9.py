# %%
import math
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x=symbols("x")

def integ(f,a,b):
    return integrate(f,(x,a,b))

print("Lab 9")
print("Bai 1")
print("a)")
print("tich phan cua f(x) = {}".format(integ(x**3+2*x**2+3,1,2)))
print("b)")
print("tich phan cua f(x) = {}".format(integ(1/x**3+1/x**2+x*sqrt(x),1,4)))
print("c)")
print("tich phan cua f(x) = {}".format(integ((x**3+x*sqrt(x)+x)/x**2,1,4)))
print("d)")
print("tich phan cua f(x) = {}".format(integ(2/x+x**3,1,2)))
print("e)")
print("tich phan cua f(x) = {}".format(integ(x**2*(1/x+2*x),1,2)))
print("f)")
print("tich phan cua f(x) = {}".format(integ((sqrt(x)-1)*(x+sqrt(x)+1),0,1)))
print("g)")
print("tich phan cua f(x) = {}".format(integ(1-2/(sin(x))**2,math.pi/4,math.pi/2)))
print("h)")
print("tich phan cua f(x) = {}".format(integ(1/(cos(x)**2+sin(x)**2),math.pi/6,math.pi/4)))
print("i)")
print("tich phan cua f(x) = {}".format(integ(exp(x)*(1-exp(-x)/cos(x)**2),0,math.pi/4)))
print("j)")
print("tich phan cua f(x) = {}".format(integ(exp(x)*(2+exp(-x)/exp(x)),0,math.pi/4)))
print("k)")
print("tich phan cua f(x) = {}".format(integ(x**2*(x-1)**2,0,1)))
print("m)")
print("tich phan cua f(x) = {}".format(integ(1/(x*(x+1)),1,2)))
print("n)")
print("tich phan cua f(x) = {}".format(integ(abs(1-x),0,2)))
print("o)")
print("tich phan cua f(x) = {}".format(integ(abs(2*x-x**2),0,2)))
print("p)")
print("tich phan cua f(x) = {}".format(integ(sqrt(x**2-3*x+2),2,4)))
print("q)")


# %%

x=symbols("x")
y=symbols("y")
def plotgraph(f,a,b,sym):
    val=np.arange(-10,10,0.1)
    f_val=lambdify(x,f,"numpy")(val)
    plt.plot(val,f_val)
    plt.title(f)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(linestyle = '-')
    plt.show()
    return integrate(f,(sym,a,b))
def plotgraph3d(f,a,b,sym):
    falam = lambdify((x,y),f,"numpy")
    xa, ya = np.meshgrid(np.arange(-1,1,0.2), np.arange(-1,1,0.1))
    za = falam(xa,ya)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = "3d")
    ax.plot_surface(xa, ya, za, cmap = plt.cm.ocean, alpha = 0.8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f)
    plt.show()
    return integrate(f,(sym,a,b))

print("Bai 2")
print("a)")
print(plotgraph(x**3-3*sin(x)*cos(x),0,pi/2,x))
print("b)")
print(plotgraph(sqrt(1+x**2+1+(x+1)**2),0,3,x))
print("c)")
print(plotgraph(sin(x**2)**2,0,1,x))
print("d)")
print(plotgraph3d(plotgraph3d(x**2*y,0,3,x),1,2,y))

# %%

x=symbols("x")
y=symbols("y")
def plotgraph(f,a,b,sym):
    val=np.arange(-10,10,0.1)
    f_val=lambdify(x,f,"numpy")(val)
    plt.plot(val,f_val)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f)
    plt.grid(linestyle = '-')
    plt.show()
    return 1/(b-a)*integrate(f,(sym,a,b))

print("Bai 3")
print("a)")
print(plotgraph(x**2-1,0,sqrt(3),x))
print("b)")
print(plotgraph(-3*x**2-1,0,1,x))
print("c)")
print(plotgraph(-x**2,0,3,x))
print("d)")
print(plotgraph(x**2-x,-2,1,x))

# %%
def plotgraph(f,a,b,sym):
    val=np.arange(-10,10,0.1)
    f_val=lambdify(x,f,"numpy")(val)
    plt.plot(val,f_val)
    plt.grid(linestyle = '-')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f)
    plt.show()
    return integrate(f,(sym,a,b))

print("Bai 4")
print("a)")
print(plotgraph(x**2*cos(x),-4,9,x))
print("b)")
print(plotgraph(exp(-1/2*x**2),-oo,oo,x))

# %%
print("Bai 5")
print("displacement of the rock: {}".format(integrate(160-32*x,(x,0,5))))

# %%
print("Bai 6")
print("c(100)-c(1)={}".format(integrate(1/(2*sqrt(x)),(x,100,1))))
print("Cost of printing posters 2-100={}".format(integrate(1/(2*sqrt(x)),(x,100,2))))

# %%
print("Bai 7")
t=symbols("t")
fa=sqrt(t+1)+5*(t**1/3)
print("tree’s height when t=0:{}".format(fa.subs(t,0)))
print("tree’s height when t=4:{}".format(fa.subs(t,4)))
print("tree’s height when t=8:{}".format(fa.subs(t,8)))
print("tree’s average height:{}".format(1/(8-0)*integrate(fa,(t,0,8))))

# %%
def plotgraph(f,a,b,n):
    val=np.arange(-3,3,0.1)
    f_val=lambdify(x,f,"numpy")(val)
    plt.plot(val,f_val)
    plt.grid(linestyle = '-')
    delta_x = (b-a)/n
    xi = [a]
    yi = [f.subs(x,a).evalf()]
    area = 0.0
    for i in range(n):
        xi.append(xi[-1] + delta_x)
        yi.append(f.subs(x,xi[-1]).evalf())
        area += yi[-1]*delta_x
    print("area = {}".format(area))
    for i in range(len(xi)):
        plt.plot([xi[i],xi[i]], [0,yi[i]], "-")
    plt.show()
def ave(f,a,b):
    return 1/(b-a)*integrate(f,(x,a,b))

print("Bai 8")
print("a)")
plotgraph(1-x,0,1,4)
plotgraph(1-x,0,1,100)
plotgraph(1-x,0,1,200)
plotgraph(1-x,0,1,1000)
print("Trung binh cua ham:{}".format(ave(1-x,0,1)))

print("b)")
plotgraph(x**2+1,0,1,4)
plotgraph(x**2+1,0,1,100)
plotgraph(x**2+1,0,1,200)
plotgraph(x**2+1,0,1,1000)
print("Trung binh cua ham:{}".format(ave(x**2+1,0,1)))

print("c)")
plotgraph(cos(x),-pi,pi,4)
plotgraph(cos(x),-pi,pi,100)
plotgraph(cos(x),-pi,pi,200)
plotgraph(cos(x),-pi,pi,1000)
print("Trung binh cua ham:{}".format(ave(cos(x),-math.pi,math.pi)))

print("d)")
plotgraph(abs(x),-1,1,4)
plotgraph(abs(x),-1,1,100)
plotgraph(abs(x),-1,1,200)
plotgraph(abs(x),-1,1,1000)
print("Trung binh cua ham:{}".format(ave(abs(x),-1,1)))

# %%
x=symbols("x")
def Trapezoidal(f,a,b,n):
    val=np.arange(a,b+0.1,0.1)
    f_val=lambdify(x,f,"numpy")(val)
    plt.plot(val,f_val)
    plt.grid(linestyle = '-')
    delta_x = (b-a)/n   
    xi = [a]
    yi = [f.subs(x,a).evalf()]
    area = 0.0
    
    for i in  range(n):
        xi.append(xi[-1] + delta_x)
        yi.append(f.subs(x,xi[-1]).evalf())
        #f(x_k-1) + f(x_k)
        area+=f.subs(x,xi[-1]).evalf()+f.subs(x,xi[-2]).evalf() 
    area*=delta_x/2
    print("area = {}".format(area))
    for i in range(len(xi)):
        plt.plot([xi[i],xi[i]], [0,yi[i]], "-")
    plt.show()


print("Bai 9")
print("a)")
Trapezoidal(exp(-x**2),0,1,3)

print("b)")
Trapezoidal(2*x**2+5*x+12,-1,5,1)
Trapezoidal(2*x**2+5*x+12,-1,5,3)
Trapezoidal(2*x**2+5*x+12,-1,5,4)
Trapezoidal(2*x**2+5*x+12,-1,5,6)

print("c)")
Trapezoidal(x**3+2*x**2-5*x-2,0,2,2)
Trapezoidal(x**3+2*x**2-5*x-2,0,2,4)
Trapezoidal(x**3+2*x**2-5*x-2,0,2,6)
Trapezoidal(x**3+2*x**2-5*x-2,0,2,8)

print("d)")
Trapezoidal(x*exp(-x),0.2,3.8,2)
Trapezoidal(x*exp(-x),0.2,3.8,4)
Trapezoidal(x*exp(-x),0.2,3.8,6)
Trapezoidal(x*exp(-x),0.2,3.8,8)


# %%
def Simpson(f,a,b,n):
    val=np.arange(a,b+0.1,0.1)
    f_val=lambdify(x,f,"numpy")(val)
    plt.plot(val,f_val)
    plt.grid(linestyle = '-')
    delta_x = (b-a)/n   
    xi = [a]
    yi = [f.subs(x,a).evalf()]
    area = 0.0
    hs=1
    for i in  range(n):
        if hs == 4:
          hs = 2
        elif hs == 2:
          hs = 4
        if i == 1:
          hs = 4
        if i == 0 or i == n-1:
          hs=1
        print(hs)
        area+=hs*(yi[-1])
        xi.append(xi[-1] + delta_x)
        yi.append(f.subs(x,xi[-1]).evalf())
        
    area*=delta_x/3
    print("area = {}".format(area))
    for i in range(len(xi)):
        plt.plot([xi[i],xi[i]], [0,yi[i]], "-")
    plt.show()

print("Bai 10")
print("a)")
Simpson(exp(-x**2),0,1,3)
print("b)")
Simpson(2*x**2+5*x+12,-1,5,1)
Simpson(2*x**2+5*x+12,-1,5,3)
Simpson(2*x**2+5*x+12,-1,5,4)
Simpson(2*x**2+5*x+12,-1,5,6)
print("c)")
Simpson(x**3+2*x**2-5*x-2,0,2,2)
Simpson(x**3+2*x**2-5*x-2,0,2,4)
Simpson(x**3+2*x**2-5*x-2,0,2,6)
Simpson(x**3+2*x**2-5*x-2,0,2,8)
print("d)")
Simpson(x*exp(-x),0.2,3.8,2)
Simpson(x*exp(-x),0.2,3.8,4)
Simpson(x*exp(-x),0.2,3.8,6)
Simpson(x*exp(-x),0.2,3.8,8)



