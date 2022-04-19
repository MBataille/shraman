import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
import pandas as pd
from matplotlib.animation import FuncAnimation

from shraman import SHRaman, params

shr = SHRaman(branch='ctest', **params)

#u0 = shr.loadInitialCondition(shr.getFilename(ext='.npy'))
shr.setInitialConditionGaussian()
shr.solve(T_transient=500, T_f=1000)
shr.saveState(-1)

v, verr = shr.getVelocity()
print(v, verr)


u0 = shr.getState(-1)
#u0 = shr.loadState(shr.getFilename(ext='.npy'))
# v = 4969

t, xs = shr.t, shr.xs

#df = pd.read_csv('../data/ctest/txs.csv')
#t, xs = df['t'], df['x']

plt.plot(t, xs)
plt.plot(t, v*t)
plt.show()



#pd.DataFrame({'t': t, 'x': xs}).to_csv('../data/ctest/txs.csv')

#print(v, verr)

plt.plot(u0)
plt.show()

X = np.append(u0, v)

shr.init_cont(u0)
du = shr.rhs_shraman(0, u0)

plt.plot(du)
plt.show()

Y = shr.cont_rhs(X)

print(f'Error is {np.abs(Y).sum()}, pinning = {Y[-1]}')

plt.plot(Y[:-1])
plt.show()

