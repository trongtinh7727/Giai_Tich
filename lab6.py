# -*- coding: utf-8 -*-
"""lab6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Oy4kZepASvaSSB48ut-JC49-sKqWxa7P
"""

from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import math

#Bai 1
print("Bai 1")
n=np.arange(1,5,1)
a=[]
b=[]
c=[]
d=[1,1]
for i in n:
  a.append(4*i+1)
  b.append(i**3)
  c.append(3**i)

n=np.arange(2,7,1)
for i in n:
  d.append(d[i-1]+d[i-2])

print("a) {}".format(a))
print("b) {}".format(b))
print("c) {}".format(c))
print("d) {}".format(d))

#bai 3
print("Bai 3")
x=symbols('x')
f=cos(x)
a=math.pi
d= f.series(x,a,6)
print("a)")
print("f(x): {}".format(d))

f=log(x)
a=2
n=np.arange(1,10,1)
d=f.series(x,a,6)
print("b)")
print("f(x): {}".format(d))

f=exp(x)
a=3
n=np.arange(1,12,1)
d=f.subs(x,a)
for i in n:
  d=d+diff(f,x,i).subs(x,a)*(x-a)**i/factorial(i)
print("c)")
print("f(x): {}".format(d))

#Bai 4
print("Bai 4")
f=cos(x)
a=0
n=np.arange(1,6,1)
d=f.subs(x,a)
for i in n:
  d=d+diff(f,x,i).subs(x,a)*(x-a)/factorial(i)
print("a)")
print("f(x): {}".format(d))


f=exp(x)
a=0
n=np.arange(1,12,1)
d=f.subs(x,a)
for i in n:
  d=d+diff(f,x,i).subs(x,a)*(x-a)**i/factorial(i)
print("b)")
print("f(x): {}".format(d))


f=1/(1-x)
a=0
n=np.arange(1,12,1)
d=f.subs(x,a)
for i in n:
  d=d+diff(f,x,i).subs(x,a)*(x-a)**i/factorial(i)
print("c)")
print("f(x): {}".format(d))

f=atan(x)
a=0
n=np.arange(1,12,1)
d=f.subs(x,a)
for i in n:
  d=d+diff(f,x,i).subs(x,a)*(x-a)**i/factorial(i)
print("c)")
print("f(x): {}".format(d))

f=(x+2)**2
a=0
n=np.arange(1,3,1)
d=f.subs(x,a)
for i in n:
  d=d+diff(f,x,i).subs(x,a)*(x-a)**i/factorial(i)
print("c)")
print("f(x): {}".format(d))

#bai 5
print("Bai 5")
n=symbols('n')
f=(4*n**2+1)/(3*n**2+2)
lf=limit(f,n,oo)
print("a) limit = {}".format(lf))

f=(n**2+1)**0.5-n
lf=limit(f,n,oo)
print("b) limit = {}".format(lf))

f=(2*n+n**0.5)**0.5-(2*n+1)**0.5
lf=limit(f,n,oo)
print("c) limit = {}".format(lf))

f=(3*5**n-2**n)/(4**n+2*5**n)
lf=limit(f,n,oo)
print("d) limit = {}".format(lf))

f=(n*sin(n**0.5))/(n**2+n-1)
lf=limit(f,n,oo)
print("e) limit = {}".format(lf))

#bai 6
print("Bai 6")
n=symbols('n', positive = True)
print("a)")
f=1-(0.2)**n
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

print("b)")
f=n**3/(n**3+1)
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

print("c)")
f=(3+5*n**2)/(n+n**2)
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

print("d)")
f=n**3/(n+1)
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

print("e)")
f=exp(1/n)
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

print("f)")
f=((n+1)/(9*n+1))**0.5
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

print("g)")
f=((-1)**(n+1)*n)/(n+n**0.5)
lf=limit(f,n,10**1700)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

print("h)")
f=tan(2*n*pi/(1+8*n))
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")
  
print("i)")
f=factorial(2*n-1)/factorial(2*n+1)
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

print("j)")
f=log(2*n**2+1)-log(n**2+1)
lf=limit(f,n,oo)
if (lf==oo):
  print("ph??n k???")
else:
  print("H???i t???")

#bai 7
print("Bai 7")
a=[]
b=[]
c=[]
d=[]
e=[1]
f=[2]
for i in range(1,5):
  a.append(1-0.2**i)
  b.append((2*i)/(i**2+1))
  c.append((-1)**(i-1)/5**i)
  d.append((1/factorial(i+1)))
  e.append(5*e[i-1]-3)
  f.append(f[i-1]/(f[i-1]+1))

print("a) {}".format(a))
print("b) {}".format(b))
print("c) {}".format(c))
print("d) {}".format(d))
print("e) {}".format(e))
print("f) {}".format(f))