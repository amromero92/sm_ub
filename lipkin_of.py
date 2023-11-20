import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import sys
import scipy.linalg as LA
from openfermion import *
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('seaborn-white')

rc('text', usetex=True)


n = 3
eps, V = 1., 1.



print('adapt LMG') 
print('parameter:t=',eps)
print('parameter:V=',V)
print('number of particles:', n)


hamiltonian_op = QubitOperator('X1 Y0', 0)
sigma_op = QubitOperator('X1 Y0', 0)

for p in range(0,n):
 termB  = QubitOperator('Z%d' %p, 1/2)

 hamiltonian_op += eps * termB
 
 for q in range(0,n):

  termA =  QubitOperator('X%d X%d' %(p, q), 1/4)
  termA -= QubitOperator('Y%d Y%d' %(p, q), 1/4)
  termA = -V * termA
  
  termS = QubitOperator('X%d Y%d' %(p, q), 1j)
  termS += QubitOperator('Y%d X%d' %(p, q), 1j)  


  hamiltonian_op += termA
  sigma_op += 1. * termS

#print(hamiltonian_op)


hamiltonian = openfermion.get_sparse_operator(hamiltonian_op)
hamil_arr = hamiltonian.toarray()

sigma = openfermion.get_sparse_operator(sigma_op)
sigma_arr = sigma.toarray()


#Diagonalize hamiltonian
wli, vli = LA.eigh(hamil_arr) 
print("Ground-state energy: ", min(wli))

h = 1
X,Y,Z = QubitOperator('X%d' % h, 1.), QubitOperator('Y%d' % h, 1.), QubitOperator('Z%d' % h, 1.)
P, M = X+1j*Y, X-1j*Y


mb_len = 2**n
mb_ref = np.zeros(mb_len)
mb_ref[-1] = 1.

print("ref state energy: ", np.dot(mb_ref.transpose(), np.dot(hamil_arr, mb_ref)).real)
 
def E_uCCD(t): 

 sigma = t*sigma_arr

 expR = scipy.linalg.expm(sigma)
 expL = scipy.linalg.expm(-sigma)
 auxmat0 = np.matmul(expL, np.matmul(hamil_arr, expR))
 energy = np.dot(mb_ref.transpose(), np.dot(auxmat0, mb_ref)).real

 return energy 


#We solve uCCD by finding optimal t  
sol_uCCD = scipy.optimize.minimize(E_uCCD, 0., method='Nelder-Mead')
euccd = E_uCCD(sol_uCCD.x)
print("sol ucc: ", sol_uCCD.x, euccd) 

tvals = np.linspace(-5,5,500)
etvals = [E_uCCD(i) for i in tvals]

commHs = hamiltonian_op*sigma_op-sigma_op*hamiltonian_op
commHs_arr = openfermion.get_sparse_operator(commHs).toarray()

ener = np.dot(mb_ref.transpose(), np.dot(hamil_arr, mb_ref)).real
commener = np.dot(mb_ref.transpose(), np.dot(commHs_arr, mb_ref)).real

nval = 16.

hadvals = [np.cos(nval*i) * ener + np.sin(nval*i)/nval * commener for i in tvals]



#-- Plot to show is periodic ------------------------------------------------
fig, ax = plt.subplots()
ax.plot(tvals, etvals,'-',label=r'$\mathrm{uCCD}$',c='g')
ax.plot(tvals, hadvals,'-',c='b')
#ax.hlines(y=-1.4142135623730925,xmin=tvals[0], xmax=tvals[-1], colors='k', linestyles='--')
#ax.hlines(y=1.4142135623730925,xmin=tvals[0], xmax=tvals[-1], colors='k', linestyles='--')
#ax.plot(tvals, etvals,'-',label=r'$\mathrm{uCCD}$',c='g')
plt.savefig('tcc_periodicity.pdf',bbox_inches='tight')  


print("here starts the proof of hadamard lemma")
print("ham: ", hamiltonian_op)
print("sig: ", sigma_op)

#print(Z*X-X*Y)
#print(Z*Y-Y*Z)

Z0 = QubitOperator('Z%d' % 0, 1.)
Z1 = QubitOperator('Z%d' % 1, 1.)
X0Y1 = QubitOperator('X%d Y%d' %(0, 1), 1.)
Y0X1 = QubitOperator('Y%d X%d' %(0, 1), 1.)
X0X1 = QubitOperator('X%d X%d' %(0, 1), 1.)
Y0Y1 = QubitOperator('Y%d Y%d' %(0, 1), 1.)


#print(Y0Y1*sigma_op-sigma_op*Y0Y1)
#sys.exit()

commHs = hamiltonian_op*sigma_op-sigma_op*hamiltonian_op



t = 2.

rhs_op = np.cos(8*t) * hamiltonian_op + np.sin(8*t)/8. * commHs
rhs = openfermion.get_sparse_operator(rhs_op).toarray()

sigma = t*sigma_arr
expR = scipy.linalg.expm(sigma)
expL = scipy.linalg.expm(-sigma)
lhs = np.matmul(expL, np.matmul(hamil_arr, expR))




#for i in range(4):
# for j in range(4):
#  print(rhs[i,j], lhs[i,j])

#sys.exit()


comm1 = hamiltonian_op*sigma_op-sigma_op*hamiltonian_op
print("comm1: ", comm1)
#sys.exit()
comm2 = comm1*sigma_op-sigma_op*comm1
print("comm2: ", comm2)
comm3 = comm2*sigma_op-sigma_op*comm2
print("comm3: ", comm3)
comm4 = comm3*sigma_op-sigma_op*comm3
print("comm4: ", comm4)
comm5 = comm4*sigma_op-sigma_op*comm4
print("comm5: ", comm5)


