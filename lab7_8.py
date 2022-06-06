# %%
import math
from sympy import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x,y=symbols('x,y')
print("LAB 7")
print("Cau 4")
def draw(f,mes):
    fa = lambdify((x,y), f,"numpy")
    xa, ya = np.meshgrid(np.arange(1.1,2,0.2),np.arange(1.1,2,0.1))
    za = fa(xa,ya)
    fig =plt.figure()
    ax=fig.add_subplot(111,projection = "3d")
    ax.plot_surface(xa, ya, za, cmap = plt.cm.ocean, alpha = 0.8)
    plt.xlabel("@x")
    plt.ylabel('@y')
    plt.title(mes)
    plt.show()
    return 0

def cau4(f):
    fxx=diff(f,x,2)+0*x
    fyy=diff(f,y,2)+0*y
    fxy=diff(f,x,1)
    fxy=diff(f,y,1)
    print("f = {}".format(f))
    mes = str(f)
    draw(f,mes)    
    print("fxx = {}".format(fxx))
    if fxx.is_real:
        print("Khong ve duoc")
    else:
        mes = str(fxx)
        draw(fxx,mes)
    print("fyy = {}".format(fyy))
    if fyy.is_real:
        print("Khong ve duoc")
    else:
        mes = str(fyy)
        draw(fyy,mes)
    print("fxy = {}".format(fxy))
    if fxy.is_real:
        print("Khong ve duoc")
    else:
        mes = str(fxy)
        draw(fxy,mes)
    return None

print("a)")
f=x+y+x*y
cau4(f)

print("b)")
f=sin(x*y)
cau4(f)

print("c)")
f=x**2*y+cos(y)+y*sin(x)
cau4(f)

print("d)")
f=x*exp(y)+y+1
cau4(f)

print("e)")
f=log(x+y)
cau4(f)

print("f)")
f=atan(y/x)
cau4(f)

print("g)")
f=x**2*tan(x*y)
cau4(f)

print("h)")
f=y*exp(x**2-y)
cau4(f)

print("i)")
f=x*sin(x**2*y)
cau4(f)

print("j)")
f=(x-y)/(x**2+y)
cau4(f)

# %%
print("Cau 5")
def checkeq(f):
    fa=diff(f,x,1)
    fra=diff(fa,y,1) 
    fb=diff(f,y,1)
    frb=diff(fb,x,1)
    return fra==frb
def printeq(check):
    if(check):
        print("fxy =  fyx")
    else:
        print("fxy != fyx")
print("Câu a")
fa=y**2*x**4*exp(x)+2
printeq(checkeq(fa))
print("Câu b")
fb=log(2*x+3*y)
printeq(checkeq(fb))
print("Câu c")
fc=x*y**2+x**2*y**3+x**3*y**4
printeq(checkeq(fc))
print("Câu d")
fd=exp(x)+x*log(y)+y*log(x)
printeq(checkeq(fd))

# %%
print("cau 6")
def cau6(f):
    f3=diff(f,x,2)
    f5=diff(f3,y,3)
    return f5
print("a)")
fa=(y**2)*(x**4)*exp(x)+2
print(cau6(fa))
print("b)")
fb=y**4+y*(sin(x)-x**4)
print(cau6(fb))
print("c)")
fc=x**5+5*x**5*y**5+sin(x)+7*exp(x)
print(cau6(fc))
print("d)")
fd=x**3*exp(y**4/2)
print(cau6(fd))

# %%
print("Cau 7")
x=symbols("x") 
y=symbols("y")
z=symbols("z")
t=symbols("t")
def cau7(w, f1, f2, f3, a): 
    w = w.subs({x:f1,y:f2,z:f3})
    dw = diff(w,t)
    return dw.subs(t,a)
print("Câu a")
print(cau7(x**2+y**2,cos(t),sin(t),0,math.pi))
print("Câu b")
print(cau7(x**2+y**2,cos(t)+sin(t),cos(t)-sin(t),0,0))
print("Câu c")
print(float(cau7(x/z+y/z,cos(t)**2,sin(t)**2,1/t,3)))
print("Câu d")
print(cau7(2*y*exp(x)-log(z),log(t**2+1),atan(t),exp(t),1))
print("Câu e")
print(cau7(z-sin(x*y),t,log(t),exp(t-1),1))


# %%
def cau8(f,x0,y0):
    dfx = (f.subs(y,y0) - f.subs({x:x0,y:y0}))/(x - x0)
    dfxdn = limit(dfx,x,x0)
    dfy = (f.subs(x,x0) - f.subs({x:x0,y:y0}))/(y - y0)
    dfydn = limit(dfy,y,y0)
    return dfxdn,dfydn
print(cau8(1-x+y-3*x**2*y,1,2))
print(cau8(4+2*x-3*y,-2,1))


# %%
def tagent(f, a, b):
    fv=f.subs({x:a,y:b})
    kx=diff(f,x,1).subs({x:a,y:b})
    ky=diff(f,y,1).subs({x:a,y:b})
    return (kx,ky)
print(tagent(2*x+3*y+4,2,-1))

# %%
print("LAB 8")
print("Cau 1")
def lab8_cau1(f):
    dif=diff(f,x,1)
    x0=solve(f,x,set = True)
    res=[]
    a=x0[1]
    for i in a:
        if i[0].is_real:
            res.append(i[0])
        
    return res
print("Câu a")
print(lab8_cau1(18*x**2-9))

print("Câu b")
print(lab8_cau1((x+2)/2*x**2))

print("Câu c")
print(lab8_cau1((-x**3/3)+(x**2)+3*x+4))

print("Câu d")
print(lab8_cau1((5*x**2+5)/x))


# %%
print("Cau 2")
def lab8_cau2(f):
    dif=diff(f,x,1)
    dif2=diff(f,x,2)
    x0=solve(f,x,set = True)
    min=[]
    max=[]
    a=x0[1]
    for i in a:
        if i[0].is_real:
            if dif2.subs(x,i[0])>0:
                min.append((i[0],f.subs(x,i[0])))
            elif dif2.subs(x,i[0])<0:
                max.append((i[0],f.subs(x,i[0])))
    print("Cac diem cuc tieu: {}".format(min))
    print("Cac diem cuc dai: {}".format(max))
    

print("Câu a")
lab8_cau2(18*x**2-9)
print("Câu b")
lab8_cau2((x+2)/(2*x**2))
print("Câu c")
lab8_cau2(-x**3/3+x**2+3*x+4)
print("Câu d")
lab8_cau2((5*x**2+5)/x)


