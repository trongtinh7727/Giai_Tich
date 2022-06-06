#%%
from sympy import * ## KHONG XOA
import numpy as np ## KHONG XOA 

global x, y, z, t ## KHONG XOA
x, y, z, t = symbols("x, y, z, t") ## KHONG XOA  
def req5(f):  ## KHONG XOA
  fx = diff(f,x,1)
  fy = diff(f,y,1)
  val = solve([fx,fy],dict = True)
  h = len(val)
  fx1 = diff(f,x,2)
  fy1 = diff(f,y,2)
  fxy = diff(fx,y,1)
  f = (fx1 * fy1) - fxy**2
  local_maxima =[]
  local_minima =[]
  saddle_point =[]
  for i in range (0,h):
    if val[i][x].is_real and val[i][y].is_real:
      m = f.subs([(x,val[i][x]),(y,val[i][y])])
      n = fx1.subs([(x,val[i][x]),(y,val[i][y])])
      if m < 0:
       saddle_point.append((val[i],[x] , val[i][y]))
      elif (m > 0 and n > 0):
       local_minima.append((val[i],[x] , val[i][y]))
      elif ( m > 0 and n < 0):
        local_maxima.append((val[i],[x] , val[i][y]))
  return local_minima,local_maxima,saddle_point

print(req5(x*y + 4))
# %%
