import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sys 

plt.style.use('seaborn-white')

rc('text', usetex=True)


N = 6

data = np.loadtxt("dataN6.dat")


x, ex, et, ev, eu, evg = [], [], [], [], [], []


for i in range(0,len(data)):
   x.append(data[i][0])
   ex.append(data[i][1])
   et.append(data[i][2])
   ev.append(data[i][3])
   eu.append(data[i][4])
   evg.append(data[i][5])         
  

errt, errv, erru, errvg = [], [], [], []

for i in range(0,len(data)):
 errt.append(abs(ex[i]-et[i])/N)
 errv.append(abs(ex[i]-ev[i])/N)
 erru.append(abs(ex[i]-eu[i])/N)
 errvg.append(abs(ex[i]-evg[i])/N)  
 
 
#-- Plot... ------------------------------------------------
fig, ax = plt.subplots()
ax.plot(x, errt,'--',label=r'$\mathrm{tCCD}$',c='b')
ax.plot(x, errv,'-.',label=r'$\mathrm{vCCD}$',c='r')
ax.plot(x, erru,'-',label=r'$\mathrm{uCCD}$',c='g')
ax.plot(x, errvg,':',label=r'$\mathrm{vgCCD}$',c='k')


ax.tick_params(direction='in', length=4, width=1, color='k', bottom=1, top=1, left=1, right=1)


ax.set_yscale('log')
plt.grid(False)



for tick in ax.get_yticklabels():
    tick.set_fontname("sans-serif")
    tick.set_fontsize(12)

ax.set_xlabel(r'$x$',fontsize=15)
ax.set_ylabel(r'$\textrm{Energy error per particle}$',fontsize=15)

ax.legend(fontsize=15)

plt.savefig('cc_lipkin_plot.pdf',bbox_inches='tight')  
