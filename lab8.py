#%%
import math
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
x=symbols("x")
def findMaxMin(f,a,b):
    df = diff(f,x,1)
    dx = solve(df,x)
    val=[f.subs(x,a),f.subs(x,b)]
    for i in dx:
        c=f.subs(x,i)
        val.append(c)
    print("Maximum:{}".format(max(val)))
    print("Minimum:{}".format(min(val)))

print("Bai 3:")
print("a)")
findMaxMin(x**3-27*x,0,5)
print("b)")
findMaxMin(3/2*x**4-4*x**3+4,0,3)
print("c)")
findMaxMin(1/2*x**4-4*x**2+5,1,3)
print("d)")
findMaxMin(5/2*x**4-20/3*x**3+6,0,3)

def drawMaxMin (f,a,b):
  try:
      root = solve(diff(f,x,1))
  except:
     root= nsolve(diff(f,x,1),x,(a+b)/2.00)
  x_root = [a,b]
  if type(root) == list:
    for r in root:
      if r.is_real:
        if r>=a and r<=b:
          x_root.append(r)
  else:
    x_root.append(root)
  x_root1 = np.unique(x_root)
  f_root = []
  for r in x_root1:
    f_root.append (f.subs (x,r))
  print(f_root)
  Abs_max_i = np.argmax(f_root) 
  Abs_min_i = np.argmin(f_root)
  print("Max = {} at x = {}".format(f_root[Abs_max_i],x_root1[Abs_max_i]))
  print("Min = {} at x = {}".format(f_root[Abs_min_i],x_root1[Abs_min_i]))
  fig = plt.figure()
  val = np.arange(a, b, 0.01)
  f_val = lambdify(x, f, "numpy")(val)
  plt.xlabel("0x")
  plt.xlabel("@y")
  plt.plot(val,f_val, "black")
  plt.plot(x_root1[Abs_max_i],f_root[Abs_max_i], "*")
  plt.plot(x_root1[Abs_min_i],f_root[Abs_min_i], "o")
  plt.show()

print("Bai 4:")
print("Câu a")
drawMaxMin(x**2-2*x-5,0,2)
#Câu b
print("Câu b")
drawMaxMin(3*x+x**3+5,-4,4)
#Câu c
print("Câu c")
drawMaxMin(sin(x)+3*x**2,-2,2)
#Câu d
print("Câu d")
drawMaxMin(exp(x**2)+3*x,-1,1)
#Câu e
print("Câu e")
drawMaxMin(x**3-3*x,-3,0)
#Câu f
print("Câu f")
drawMaxMin(x**3-3*x,0,3)

def goldensearch(f,a,b,e,check):
    d=b-a
    val=np.arange(a,b,0.01)
    f_val=lambdify(x,f,"numpy")(val)
    plt.plot(val,f_val)
    while True:
        if b-a<e:
            break
        d=0.618*d
        x1=b-d
        x2=a+d
        if f.subs(x,x1)<=f.subs(x,x2):
            b=x2
        else:
            a=x1
        min=a
        if(check == 1):
          plt.plot(min,f.subs(x,min),"ro")
        
    if (check != 1):
      plt.plot(min,f.subs(x,min),"ro")
    plt.grid(linestyle = '-')
    plt.show()
print("Bai 5")
goldensearch(x**2,-2,1,0.3,1)

def fibosearch(f,a,b,e,check):
    fibo=[2,3]
    n=2
    val=np.arange(a,b,0.1)
    f_val=lambdify(x,f,"numpy")(val)
    plt.plot(val,f_val)
    while True:
        if b-a<e:
            break
        d=b-a
        x1=b-d*(fibo[n-2]/fibo[n-1])
        x2=a+d*(fibo[n-2]/fibo[n-1])
        if f.subs(x,x1)<=f.subs(x,x2):
            b=x2
        else:
            a=x1
        min=a
        if(check == 1): 
            plt.plot(min,f.subs(x,min),"ro")
            fibo.append(fibo[n-2]+fibo[n-1])
            n=n+1
    if(check != 1):
      plt.plot(min,f.subs(x,min),"ro")

    plt.grid(linestyle = '-')
    plt.show()
fibosearch(x**2,-2,1,0.3,1)

m=symbols("m")
def DeterM(f,xo):
    fo=diff(f,x,1).subs(x,xo)
    dm=solve(fo,m)
    max=[]
    for i in dm:
        df=f.subs(m,i)
        if diff(df,x,1).subs(x,i)<0:
            max.append(i)
    return max
print(DeterM(x**3-3*m*x**2+3*(m**2-1)*x-m**2-1,1))

print("Bai 8")
print("a)")
goldensearch(-2*x**2+x+4,-5,5,1/9,0)
fibosearch(-2*x**2+x+4,-5,5,1/9,0)

print("b)")
goldensearch(-4*x**2+2*x+2,-6,6,1/10,0)
fibosearch(-4*x**2+2*x+2,-6,6,1/10,0)

print("c)")
goldensearch(x**3+6*x**2+5*x-12,-5,2,1/10,0)
fibosearch(x**3+6*x**2+5*x-12,-5,2,1/10,0)

print("d)")
goldensearch(2*x-x**2,0,3,1/100,0)
fibosearch(2*x-x**2,0,3,1/100,0)

print("e)")
goldensearch(x**2-x-10,-10,10,1/5,0)
fibosearch(x**2-x-10,-10,10,1/5,0)
# %%
