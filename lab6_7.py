#%%
import math
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

print("Cau 7")

print("Cau a")
n=symbols("n")
fa=1-(-2/exp(1))**n
val=np.arange(-40,40)
yval =lambdify(n,fa,"numpy") (val)
plt.plot(val,yval)
plt.show()

print("Cau b")
n=symbols("n")
fb=sqrt(n)*sin(pi/sqrt(n))
val=np.arange(1,40)
yval =lambdify(n,fb,"numpy") (val)
plt.plot(val,yval)
plt.show()

print("Cau c")
n=symbols("n")
fc=sqrt((3+n**2)/(8*n**2+n))
val=np.arange(-40,40,1.5)
yval =lambdify(n,fc,"numpy") (val)
plt.plot(val,yval)
plt.show()

print("Cau d")
n=symbols("n")
fd=(n**2*cos(n))/(1+n**2)
val=np.arange(-40,40)
yval =lambdify(n,fd,"numpy") (val)
plt.plot(val,yval)
plt.show()

def sum(n):
    ni=np.arange(1,n+1,1)
    s=1
    for i in ni:
        s=s*(2*i-1)
    return s
print("Cau e")
val=np.arange(1,10)
yn=[]
for i in range(1,10,1):
    yn.append(sum(i)/factorial(i))
plt.plot(val,yn,"-o")
plt.show()

print("Cau f")
yf=[]
for i in range(1,10,1):
    yf.append(sum(i)/((2*i)**i))
plt.plot(val,yf,"m")
plt.show()

# %%

print("Cau 8")
print("Câu a")
x=symbols("x")
fa=exp(x)
fca=fa.subs(x,0)
for i in range(0,4,1):
    fta=(diff(fa,x,i).subs(x,0)/math.factorial(i))*(x-0)**i
    fca=fca+fta
    print("bac {}: {}".format(i,fca))


print("Câu b")
x=symbols("x")
fb=sin(x)
fbc=fb.subs(x,0)
for i in range(0,4,1):
    ftc=(diff(fb,x,i).subs(x,0)/math.factorial(i))*(x-0)**i
    fbc=fbc+ftc
    print("bac {}: {}".format(i,fbc))


print("Câu c")
x=symbols("x")
fc=log(x)
fcc=fc.subs(x,1)
for i in range(0,4,1):
    ftc=(diff(fc,x,i).subs(x,1)/math.factorial(i))*(x-1)**i
    fcc=fcc+ftc
    print("bac {}: {}".format(i,fcc))

print("Câu d")
x=symbols("x")
fd=1/x
fdc=fd.subs(x,2)
for i in range(0,4,1):
    ftd=(diff(fd,x,i).subs(x,2)/math.factorial(i))*(x-2)**i
    fdc=fdc+ftd
    print("bac {}: {}".format(i,fdc))


# %%
print("Cau 9")

print("Câu a")
fa = [1,1]
n = 8
b = 1
c = 2 
for i in range(n-2) :
  fa.append(fa[-1]+fa[-2])
print ("{}".format(fa[-1]))
print("Câu b")
phi = 1.618034
i = 8
xb = round((phi**i - (1-phi)**i)/sqrt(5),0)
print ("{}".format(xb))

print("Câu c")
fc = [1,1]
phi = 1.618034
n = 8
for i in range (n-2) :
    fc.append(round(fc[-1]*phi,0))
print ("{}".format(fc[-1]))

# %%
print("Câu 10")
print("a")
x = 0
y = 0
d = 2
alpha = 90
alpha_c = 0
F = "F + F - F - F + F"
r0 = F
#doi tu do sang radian
d2r = math.pi/180
n = 5
F_new = F
j = 0

while j<n:
  F_new = F.replace("F",r0)  #tai diem co gt F thì thay = r0
  F = F_new
  j = j+1
Px = [x]
Py = [y]

for i in F:
  if i == "F":
    x = x - d*math.cos(alpha_c*d2r)
    y = y - d*math.sin(alpha_c*d2r)
    Px.append(x)
    Py.append(y)
  elif i == "+":
    alpha_c = alpha_c + alpha
  elif i == "-":
    alpha_c = alpha_c - alpha
    
fg = plt.figure()
plt.plot(Px,Py,'r')
plt.show()

#[ : dùng 1 mảng, lưu lại giá trị tọa độ điểm hiện tại vào trong mảng
#] : lấy tọa độ điểm được lưu trước đó trong mảng, thay thế cho tọa độ điểm hiện tại

print("b")
x = 0
y = 0
d = 2
alpha = 45
alpha_c = 0
F = "F[+F][-F]"
r0 = F
#doi tu do sang radian
d2r = math.pi/180
n = 4
F_new = F
j = 0

while j<n:
  F_new = F.replace("F",r0)  #tai diem co gt F thì thay = r0
  F = F_new
  j = j+1
Px = [x]
Py = [y]
xt=[]
yt=[]

for i in F:
  if i == "F":
    x = x - d*math.cos(alpha_c*d2r)
    y = y - d*math.sin(alpha_c*d2r)
    Px.append(x)
    Py.append(y)
  elif i == "+":
    alpha_c = alpha_c + alpha
  elif i == "-":
    alpha_c = alpha_c - alpha
  #[ : dùng 1 mảng, lưu lại giá trị tọa độ điểm hiện tại vào trong mảng
  elif i == "[":
      xt.append(Px[-1])
      yt.append(Py[-1])
  #] : lấy tọa độ điểm được lưu trước đó trong mảng, thay thế cho tọa độ điểm hiện tại
  elif i == "]":
      Px[-1]=xt[-1]
      Py[-1]=yt[-1]
fg = plt.figure()
plt.plot(Px,Py,'r')
plt.show()

# %%
print("Lap 7")

print("Câu 1:")

print("a)")
x, y = symbols('x, y')
fa = x**2 + x*y**3
xy=([0,0],[-1,1],[2,3],[-3,2])
for i in xy:
    res = fa.subs({x:i[0],y:i[1]})
    print("f({},{}) = {}".format(i[0],i[1],res))

print("b)")
x, y,z = symbols('x, y,z')
fb = (x-y)/(y**2+z**2)
xy=([3,-1,2],[1,0.5,1/4.0],[0,-1/3,0],[2,2,100])
for i in xy:
    res = fa.subs({x:i[0],y:i[1],z:i[2]})
    print("f({},{},{}) = {}".format(i[0],i[1],i[2],res))


# %%
from mpl_toolkits.mplot3d import Axes3D
print("Cau 2")
x, y = symbols('x, y')
def draw(f):
    mes=str(f)
    fa = lambdify((x,y), f,"numpy")
    xa, ya = np.meshgrid(np.arange(-1,1,0.2),np.arange(-1,1,0.1))
    za = fa(xa,ya)
    fig =plt.figure()
    ax=fig.add_subplot(111,projection = "3d")
    ax.plot_surface(xa, ya, za, cmap = plt.cm.ocean, alpha = 0.8)
    plt.xlabel("@x")
    plt.ylabel('@y')
    plt.title(mes)
    plt.show()
    return 0

print("a)")
f=cos(x)*cos(y)*exp(-sqrt(x**2+y**2)/4)
draw(f)

print("b)")
f=-(x*y**2)/(x**2+y**2)
draw(f)

print("c)")
f=(x*y*(x**2-y**2)/(x**2+y**2))
draw(f)

print("d)")
f=y**2-y**4-x**2
draw(f)


# %%
print("Cau 3")
x,y=symbols('x,y')
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

def cau3(f):
    fx=diff(f,x,1)+0*x
    fy=diff(f,y,1)+0*y
  
    print("f = {}".format(f))
    mes = str(f)
    draw(f,mes)    
    print("fx = {}".format(fx))
    if fx.is_real:
        print("Khong ve duoc")
    else:
        mes = str(fx)
        draw(fx,mes)
    print("fy = {}".format(fy))
    if fy.is_real:
        print("Khong ve duoc")
    else:
        mes = str(fy)
        draw(fy,mes)
   
    return None

print("a)")
f=2*x**2-3*y-4
cau3(f)

print("b)")
f=(x**2-1)*(y+2)
cau3(f)

print("c)")
f=x**2-x*y+y**2
cau3(f)

print("d)")
f=5*x*y-7*x**2-y**2+3*x-6*y+2
cau3(f)

print("e)")
f=(x*y-1)**2
cau3(f)

print("f)")
f=(x*2-3*y)**3
cau3(f)

print("g)")
f=(x**2+y**2)**0.5
cau3(f)

print("h)")
f=(x**3+y/2)**(2/3)
cau3(f)

print("i)")
f=1/(x+y)
cau3(f)

print("j)")
f=x/(x**2+y**2)
cau3(f)

print("k)")
f=(x+y)/(x*y-1)
cau3(f)

print("l)")
f=atan(y/x)
cau3(f)

print("m)")
f=exp(x+y+1)
cau3(f)

print("n)")
f=exp(-x)*sin(x+y)
cau3(f)

print("o)")
f=log(x+y)
cau3(f)


