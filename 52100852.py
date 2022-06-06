from sympy import * 
import numpy as np  

global x, y, z, t 
x, y, z, t = symbols("x, y, z, t")      

def req1(f, g, a):
    f=f+0*x
    g=g+0*x

    m=diff(f+g,x,1).subs(x,a)
    if m.has(zoo,nan) or m == None:
      m = None
    else:
      m=round(float(m),2)

    n=diff(f*g,x,1).subs(x,a)
    if n.has(zoo,nan) or n == None:
      n = None
    else:
      n=round(float(n),2)

    p=diff(f.subs(x,g),x,1).subs(x,a)
    if p.has(zoo,nan) or p == None or f.subs(x,g).has(zoo,nan) :
      p = None
    else:
      p=round(float(p),2)

    q=diff(f/g,x,1).subs(x,a)
    if q.has(zoo,nan) or q == None or (f/g).has(zoo,nan):
      q = None
    else:
      q=round(float(q),2)
    tup=(m,n,p,q)
    return tup

def req2(f, a, b, c):  
    f=f+0*x
    try:
      fabc=float(f.subs([(x,a),(y,b),(z,c)]))
      fxabc=float(diff(f,x,1).subs([(x,a),(y,b),(z,c)]))
      fyabc=float(diff(f,y,1).subs([(x,a),(y,b),(z,c)]))
      fzabc=float(diff(f,z,1).subs([(x,a),(y,b),(z,c)]))
      result= fabc +fxabc*(x-a)+fyabc*(y-b)+fzabc*(z-c)
      if result.has(nan,zoo) or result.is_real :
        return None
      return result
    except:
      return None

def req3(w, f1, f2, f3, a):
    w=w+0*x  
    d=w.subs([(x, f1), (y, f2),(z,f3)])
    result=diff(d,t,1).subs(t,a)
    if  result.has(nan,zoo) or d.has(nan,zoo): 
      return None
    return float(result) 

def req4(a, b, n):  
  d=0
  for k in range (0,n+1):
    hs=float(factorial(n)/(factorial(k)*factorial(n-k)))
    d=d+hs*(b**k)*a**(n-k)
  return d

def req5(f):  
    fx=diff(f,x,1)
    fy=diff(f,y,1)
    x0=solve([fx,fy],[x,y],set =True)
    x1=x0[1]
    d=diff(f,x,2)*diff(f,y,2)-diff(fy,x,1)**2
    res=([],[],[])
    for i in x1:
      if i[0].is_real and i[1].is_real:
        a=d.subs([(x,i[0]),(y,i[1])])
        if a>0 and diff(f,x,2).subs([(x,i[0]),(y,i[1])])>0:
          res[0].append(i)
        if a>0 and diff(f,x,2).subs([(x,i[0]),(y,i[1])])<0:
          res[1].append(i)
        if a<0:
          res[2].append(i)   
    return res

def req6(message, x, y, z):  
    re=""
    key=abs(x**2-y**2-z)
    for i in message:
      r= key^ord(i)
      re=re+chr(r)
    return re


def req7(xp, yp, xs):
  n=0
  for i in xp:
    n+=1
  xk=0
  yk=0
  xy=0
  x2=0
  for i in xp:
    xk=xk+i
  for i in yp:
    yk=yk+i
  for i in range(n):
    xy=xy+xp[i]*yp[i]
  for i in xp:
    x2=x2+i**2
  m=(xk*yk-n*xy)/(xk**2-n*x2)
  b=(1/n)*(yk-m*xk)
  res=m*xs+b
  return round(float(res),2)

def req8(f, eta, xi, tol): 
    df=diff(f,x,1)
    x0=[xi]
    i=0
    x_new = x0[-1] - eta*df.subs(x,x0[-1])
    while abs(df.subs(x,x_new)) >= tol:
        i=i+1
        if i > 700:
          return None
        x_new = x0[-1] - eta*df.subs(x,x0[-1])
        if x_new.has(nan,zoo):
          return None  
        x0.append(float(x_new))
    if x_new.has(nan,zoo):
          return None  
    return round(float(x0[-1]),2)