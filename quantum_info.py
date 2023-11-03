#Script to compute some quantum-information tools within the nuclear shell model
#Important: Jordan-Wigner mapping is always supposed to be the fermionic mapping implemented, where each orbital in the 
#configuration space corresponds to a qubit, which can be empty (projection 0) or occupied (projection 1)


import sys
import numpy as np
import numpy.linalg as LA
import itertools as it
import scipy.sparse
from qibo import quantum_info
from qibo.backends.numpy import NumpyBackend
npe = NumpyBackend()


sp_len          = 6                      #Dimension of configuration space (only for one fluid, protons or neutrons)
neu_num,pro_num = 2,2                    #Number of neutrons and protons 
statefile       = "ground_state.dat"     #State to read
mbfile          = "many_body_states.dat" #Many-body basis to read

a=neu_num+pro_num
if (neu_num == 0 or pro_num == 0):
 nqubits=sp_len
else:
 nqubits=2*sp_len
 
 
#Pauli matrices
iden    = scipy.sparse.identity(2, format='csc', dtype=complex)
pauli_x = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
pauli_y = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
pauli_z = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex) 


#Open many-body states
slaters = []
f = open(mbfile,'r')
dim_M = int(f.readline())
for i in range(dim_M):
 mel = f.readline().strip().split() 
 slt = []
 for j in range(1,len(mel)):
  ttmel = mel[j]
  if j==1:
   ttmel = ttmel.replace("(","")
   ttmel = ttmel.replace(",", "")
  elif j==len(mel)-1:
   ttmel = ttmel.replace(")","")
  else:
   ttmel = ttmel.replace(",", "")
  slt.append(int(ttmel))
 slaters.append(tuple(slt))
f.close()


#Open ground state (amplitudes, in the slater determinants basis)
amplitudes = [0.]*dim_M
f = open(statefile,'r')
len_amp = len( f.readlines() )
f.close()
f = open(statefile,'r')
for i in range(len_amp):
 mel = f.readline().strip().split() 
 amplitudes[int(mel[0])] = float(mel[1])
f.close()

#Check dimensions are the same
assert len(slaters) == len(amplitudes)


#We get the qubit representation of the many-body slater determinants, 1 if state is occupied and 0 if not (JW)
#For example, state (1,3,5,8) of four particles in a configuration space of 10 states is given by: '01010100100'
qubit_slaters = []
for i in slaters: 
 quantum_state = [0]*nqubits 
 for j in range(len(i)):
  quantum_state[i[j]] = 1
 qubit_slaters.append(''.join([str(n) for n in quantum_state]))
 


#Given a qubit state of the form '01010100100', builds a column vector with a one in the corresponding index in 2^nqubit space
#Assumes that projection 0 is (1,0) and projection 1 is (0,1)
#For example: 00 is (1,0,0,0), 01 is (0,1,0,0), 10 is (0,0,1,0) and 11 is (0,0,0,1)
def comp_state_from_qubit_state(qubitstate):

 l = len(qubitstate)
 
 q0, q1 = scipy.sparse.csc_matrix([1,0]), scipy.sparse.csc_matrix([0,1])

 
 qb1, qb2 = int(qubitstate[0]), int(qubitstate[1])
 if (qb1 == 0):
  if (qb2 == 0):
   cs = scipy.sparse.kron(q0,q0)
  else:
   cs = scipy.sparse.kron(q0,q1)
 else:
  if (qb2 == 0):
   cs = scipy.sparse.kron(q1,q0)
  else:
   cs = scipy.sparse.kron(q1,q1)
   
 for i in range(2,l):
  qb = int(qubitstate[i])
  if (qb == 0):
   cs = scipy.sparse.kron(cs, q0)
  else: 
   cs = scipy.sparse.kron(cs, q1)   
 
 return cs.toarray()
 


#Get state in computational basis where each index is a different qubit state.
#It is written in the basis of Slater determinants in the qubit representation
#Multiplied by the amplitudes of the ground state (or any other state)
comp_state = np.zeros((1,2**nqubits))
for i in range(len(slaters)):
 comp_state += amplitudes[i]*comp_state_from_qubit_state(qubit_slaters[i])


#Von Neumann entropy given a reduced density matrix
def entropy(red_rho):
 w,v = LA.eig(red_rho)
 entropy = 0.
 for rho_i in w:
  if (abs(rho_i) > 1e-8):
   entropy -= rho_i * np.log2(rho_i)
 assert np.isclose(entropy.imag, 0.)
 
 return entropy.real


#Function to compute the mutual information between orbitals i and j within a wavefunction "comp_state"
#It is assumed that "comp_state" is already computed somewhere and nqubits is given too 
def mutual_info(i,j):

 rho_i = npe.partial_trace(comp_state, [i], nqubits) 
 rho_j = npe.partial_trace(comp_state, [j], nqubits) 
 rho_ij = npe.partial_trace(comp_state, [i,j], nqubits)   
 
 S_i = entropy(rho_i)
 S_j = entropy(rho_j)
 S_ij = entropy(rho_ij)
 return .5*(S_i+S_j-S_ij)

#Single orbital entropy with occupation number as input
def Si(occ):
 return -occ * np.log2(occ) - (1.-occ)*np.log2(1.-occ)
 
#Function to check if state is pure: Tr(rho^2) = 1
def pure_state(state):
 rho = state.T.dot(state)
 rhosq = rho.dot(rho)
 if (abs(np.trace(rhosq)-1.) < 1e-8): return True
 return False  
 
#Get the statevector that builds up rho as rho=state.T.dot(state) 
#Important: rho must be idempotent and symmetric (a pure state)
def statevector_rho(rho): 
 U,S,Vh = LA.svd(rho)
 statevector = U[:,0]
 return statevector.reshape(1,rho.shape[0])
 
#Concurrence formula according to Robin in Sec 3.2.2 Eur. Phys. J. A 59, 231 (2023)
#partition is a list of single-particle orbitals
def concurrence(partition,state):
  
 rho = state.T.dot(state) 
 rho_A = npe.partial_trace(state, partition, nqubits)  
 rho_A_sq = rho_A.dot(rho_A)
  
 C_AB = 2.*(1.-np.trace(rho_A_sq))

 return C_AB

#Function to compute n-tangles according to Eq. (37) Eur. Phys. J. A 59, 231 (2023)
#It is always assumed that the state is real
def n_tangle(qubit_list, nqubits, state):

 if len(qubit_list) == 1: 
  print("1-tangles do not make sense...")
  sys.exit()

 #We build the operator first
 ops_i = [0]*nqubits
 for p in qubit_list:
  ops_i[p] = 1
  
 
 if (ops_i[0] == 0):
  if (ops_i[1] == 0):
   ntangle_op = scipy.sparse.kron(iden,iden)
  else:
   ntangle_op = scipy.sparse.kron(iden,pauli_y)
 else:
  if (ops_i[1] == 0):
   ntangle_op = scipy.sparse.kron(pauli_y,iden)
  else:
   ntangle_op = scipy.sparse.kron(pauli_y,pauli_y)
      
 for i in range(2,nqubits):
  if (ops_i[i] == 0):
   ntangle_op = scipy.sparse.kron(ntangle_op, iden)
  else: 
   ntangle_op = scipy.sparse.kron(ntangle_op, pauli_y)  
 
  
 #Now we apply it to the state, basic matrix multiplication
 ket = state.T
 bra = ket.T
 ntangle = abs(bra.dot(ntangle_op.dot(ket)))**2
 
 return ntangle.item() 
 

#START
#Generate all nonzero ntangles with all possible combinations of orbitals and write to file

#Orbitals 
sp_orbitals = [i for i in range(nqubits)]

#Ground state in computational basis. Copy
many_body_state = comp_state[:]
many_body_state_qibo = many_body_state.reshape(2**nqubits)


#Pure state?
print("Is the state pure: ", pure_state(many_body_state))


"""
#Print mutual info to file
f = open('mutual_info.dat','w')
f.write("orb1 orb2 S12\n")
#f.write("---------------------------------------------\n")
for i in range(nqubits):
 for j in range(i,nqubits):

  if (i == j):
   f.write(str(i) + " " + str(j) + " " + str(0.) + "\n")
  else:
   sij = mutual_info(i,j)
   f.write(str(i) + " " + str(j) + " " + str(sij) + "\n")
   f.write(str(j) + " " + str(i) + " " + str(sij) + "\n")   

f.close()


#Print n-tangles to file
f = open('ntangles.dat','w')
f.write("n     orbitals                           n-tangle\n")
f.write("---------------------------------------------\n")
for n_sel in range(2,nqubits,2):
 orb_sel = list(it.combinations(sp_orbitals, n_sel)  )
 for orbs in orb_sel:
  lorbs = list(orbs)
  nt = n_tangle(lorbs, nqubits, gs_cb)
  if (nt > 0.): 
   f.write(str(n_sel) + "   ")
   for o in lorbs: f.write(str(o) + "    ")
   f.write(str(nt) + "\n")
f.close()
"""



### TESTS


"""
#Test B.1 Eur. Phys. J. A 59, 231 (2023) 
a1 = comp_state_from_qubit_state('11110')
a2 = comp_state_from_qubit_state('00000')
a3 = comp_state_from_qubit_state('10101') 
cstate = 1./np.sqrt(3.) * (a1+a2+a3)

orb_sel = list(it.combinations([0,1,2,3,4], 4)  )
for orbs in orb_sel:
 lorbs = list(orbs)
 nt = n_tangle(lorbs, 5, cstate)
 if (nt > 0.): 
  print(list(orbs), nt)
   
   
#print(n_tangle([1,0,2,3], 5, cstate))
#print(n_tangle([0,1,3,2], 5, cstate))
#print(n_tangle([0,3,1,2], 5, cstate))

#End of test B.1
"""









