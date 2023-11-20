import sys
import numpy as np
import numpy.linalg as la
import itertools as it
import scipy.optimize 
import scipy.linalg
import copy 
sys.getdefaultencoding()

## Implements example of J. Chem Phys 148 044107 (2018)





#If true, prints info about states of the Lipkin model
verbose = True

#Number of particles and value of x in Hamiltonian (4)
N, x = 6, 0.55

#Number of levels is twice the number of particles (two N-fold degenerate levels)
p_num = 2 * N

#Number of single-particle states 
sp_len = p_num

#Number of many-body states 
#mb_len = np.math.factorial(sp_len)/(np.math.factorial(N) * np.math.factorial(sp_len-N))

#Build single-particle states and table
sp_states = []
sp_states_table = []
for p in range(1,N+1):
  for sigma in range(-1,2,2):
     sp_states.append([p,sigma])

for i in range(0,len(sp_states)):
   sp_states_table.append([i+1,sp_states[i]])


#Build N-particle in the Lipkin model. 
mb_states = []
mb_states_table = []
sigmas = list(it.product([-1,1],repeat=N))  #All the possible combinations of spins. It has to be 2^N
for i in sigmas:
  mb_state = []
  for j in range(0,N):
    mb_state.append([j+1,i[j]])
  mb_states.append(tuple(mb_state))   #We append the level and the spin of the particle in the level
  
  
mb_len = len(mb_states)

assert mb_len == 2**N  

for i in range(0,len(mb_states)):
   mb_states_table.append([i+1,mb_states[i]])

####################################################


#Print the single-particle and many-body states 

if (verbose == True):

 print('!----------Single-particle states-----------!')
 print('  ')
 #print('i,|p,s>')
 for i in range(0,len(sp_states)):
   print(sp_states_table[i][0],sp_states_table[i][1])
 print(' ')
 print('  ')

 print('!----------Many-body states-----------!')
 print('  ')
 for i in range(0,len(mb_states)):
   print(mb_states_table[i][0],mb_states_table[i][1])
 print(' ')



#Applies the operator J_+ J_+ on a many-body state. Flips two spins from - to + and returns all the resulting states
def JpJp(mb_state): 
 flipped_spin_states = [] 
 for s in range(len(mb_state)):
  for r in range(s+1,len(mb_state)):
   new_state = copy.deepcopy(list(mb_state))  #Important to use this, otherwise mb_state is modified
   sp_s, sp_r = mb_state[s], mb_state[r]
   if (sp_s[1] == -1 and sp_r[1] == -1):      #Spins down
    new_state[s][1], new_state[r][1] = 1, 1   #Flip to spin up
    flipped_spin_states.append(tuple(new_state))
    
 return flipped_spin_states 
 
#Applies the operator J_- J_- on a many-body state. Flips two spins from + to - and returns all the resulting states
def JmJm(mb_state): 
 flipped_spin_states = [] 
 for s in range(len(mb_state)):
  for r in range(s+1,len(mb_state)):
   new_state = copy.deepcopy(list(mb_state))  #Important to use this, otherwise mb_state is modified
   sp_s, sp_r = mb_state[s], mb_state[r]
   if (sp_s[1] == 1 and sp_r[1] == 1):        #Spins up
    new_state[s][1], new_state[r][1] = -1, -1 #Flip to spin down
    flipped_spin_states.append(tuple(new_state))
    
 return flipped_spin_states  
 
#Applies the operator J_+ J_- on a many-body state. Flips two spins and returns all the resulting states
def JpJm(mb_state): 
 flipped_spin_states = [] 
 for s in range(len(mb_state)):
  for r in range(len(mb_state)):
   new_state = copy.deepcopy(list(mb_state))  #Important to use this, otherwise mb_state is modified
   sp_s, sp_r = mb_state[s], mb_state[r]
   if (sp_s[1] == 1 and sp_r[1] == -1):        #Spins up and down
    new_state[s][1], new_state[r][1] = -1, 1  #Flips the spins
    flipped_spin_states.append(tuple(new_state))
    
 return flipped_spin_states   
 
#Applies the operator J_z on a many-body state. Counts the number of particles in upper and lower shells and returns the difference over 2
def Jz(mb_state):
 M = 0.
 for i in mb_state:
  M += i[1]
 M = 0.5 * M
 return M
 
#Kronecker delta function between any elements a and b (numbers, vectors...)
def kron_delta(a,b): 
  kron_delta  = 0
  if (a == b):
    kron_delta =  1
  return kron_delta
  
def check_symmetric(a, rtol=1e-05, atol=1e-08):
  return np.allclose(a, a.T, rtol=rtol, atol=atol)



#Build Hamiltonian
hamil = np.zeros([mb_len,mb_len])

#One-body part
for s in range(mb_len):
 mbs = mb_states[s]
 hamil[s,s] = x * Jz(mbs)
 
#Two-body part
for s in range(mb_len):
 mbs = mb_states[s]
 for r in range(s+1,mb_len): 
  mbr = mb_states[r]
  states_pp = JpJp(mbr)
  states_mm = JmJm(mbr)
  for spp in states_pp:
   hamil[s,r] += kron_delta(mbs,spp)
  for smm in states_mm:
   hamil[s,r] += kron_delta(mbs,smm)
  hamil[s,r] = -(1.-x)/N * hamil[s,r]  
  hamil[r,s] = hamil[s,r]    
 

assert check_symmetric(hamil) == True  

if (verbose == True): 
 print('!----------Hamiltonian-----------!')
 print('  ')
 for i in range(mb_len):
  for j in range(i,mb_len):
   print(i,j,hamil[i,j])
 print(' ')  


#Diagonalize hamiltonian
wli, vli = la.eig(hamil) 
 
print("LIPKIN MODEL")
print("-------------------")
print("Number of particles: ", N)
print("Interaction x: ", x)
print("Ground-state energy: ", min(wli))
print("-------------------")
print("Hamiltonian (4) from J. Chem Phys 148 044107 (2018) ")
print(" ")
#sys.exit("LIPKIN MODEL DONE")



## Now we start with the UCC approximations 

#Reference state is the Slater determinant with all spins down. In our implementation of the Fock basis, it corresponds to the first state
mb_ref = np.zeros(mb_len)
mb_ref[0] = 1

#We build the J_+ J_+ cluster matrix 
JpJp_cc = np.zeros([mb_len,mb_len])
for s in range(mb_len):
 for r in range(mb_len):
  mbs, mbr = mb_states[s], mb_states[r]
  states_pp = JpJp(mbr)
  for spp in states_pp:
   JpJp_cc[s,r] += kron_delta(mbs,spp)
   
#We build the J_- J_- cluster matrix 
JmJm_cc = np.zeros([mb_len,mb_len])
for s in range(mb_len):
 for r in range(mb_len):
  mbs, mbr = mb_states[s], mb_states[r]
  states_mm = JmJm(mbr)
  for smm in states_mm:
   JmJm_cc[s,r] += kron_delta(mbs,smm)
   
#We build the J_+ J_- cluster matrix 
JpJm_cc = np.zeros([mb_len,mb_len])
for s in range(mb_len):
 for r in range(mb_len):
  mbs, mbr = mb_states[s], mb_states[r]
  states_pm = JpJm(mbr)
  for spm in states_pm:
   JpJm_cc[s,r] += kron_delta(mbs,spm)   
   
   

## Traditional coupled cluster approach 

print("Traditional CC")
print("-------------------")

# Condition of Eq. (18) to find the optimal value of t
def tCCD(t): 
 expR = scipy.linalg.expm(t*JpJp_cc)
 expL = scipy.linalg.expm(-t*JpJp_cc)
 auxmat0 = np.matmul(expL, np.matmul(hamil, expR))
 auxmat1 = np.matmul(JmJm_cc, auxmat0) 
 cc_val = np.dot(mb_ref.transpose(), np.dot(auxmat1, mb_ref))
 
 return cc_val 
 
## Energy for the traditional CC, eq. (16)
def E_tCCD(t):
 expR = scipy.linalg.expm(t*JpJp_cc)
 expL = scipy.linalg.expm(-t*JpJp_cc)
 auxmat = np.matmul(expL, np.matmul(hamil, expR))
 energy = np.dot(mb_ref.transpose(), np.dot(auxmat, mb_ref))
 
 return energy
 
#We solve tCCD by finding t such that tCCD(t) = 0 
sol_tCCD = scipy.optimize.fsolve(tCCD, 0.)

etccd = E_tCCD(sol_tCCD)

#Print optimal t and corresponding CC energy
print(sol_tCCD,etccd)  

## End of traditional coupled cluster approach


## Extended coupled cluster approach 

print("Extended CC")
print("-------------------")

# Energy of Eq. (20) 
def E_eCCD(args):
 t = args[0]
 z = args[1] 
 expR = scipy.linalg.expm(t*JpJp_cc)
 expL = scipy.linalg.expm(-t*JpJp_cc)
 expL2 = scipy.linalg.expm(z*JmJm_cc)
 auxmat0 = np.matmul(expL, np.matmul(hamil, expR))
 auxmat1 = np.matmul(expL2, auxmat0) 
 energy = np.dot(mb_ref.transpose(), np.dot(auxmat1, mb_ref))
 
 return energy

#NOT SURE IF THIS IS CORRECT. t and z are found by making the energy stationary...
#We solve eCCD by finding optimal t,z  
#sol_eCCD = scipy.optimize.minimize(E_eCCD, [0.,-0.], method='Nelder-Mead')
#print(sol_eCCD)
#print(E_eCCD([0.23,0.]))

## End of extended coupled cluster approach 


## Variational coupled cluster approach 

print("Variational CC")
print("-------------------")

# Energy of Eq. (21) 
def E_vCCD(t): 
 expR = scipy.linalg.expm(t*JpJp_cc)
 expL = scipy.linalg.expm(t*JmJm_cc)
 auxmat0 = np.matmul(expL, np.matmul(hamil, expR))
 auxmat1 = np.matmul(expL, expR)
 norm =  np.dot(mb_ref.transpose(), np.dot(auxmat1, mb_ref))
 energy = np.dot(mb_ref.transpose(), np.dot(auxmat0, mb_ref))
 
 energy = energy/norm
 
 return energy
 
 
#We solve vCCD by finding optimal t  
sol_vCCD = scipy.optimize.minimize(E_vCCD, 0., method='Nelder-Mead')

evccd = E_vCCD(sol_vCCD.x)
print(sol_vCCD.x, evccd) 



## End of variational coupled cluster approach 

## Unitary coupled cluster approach 

print("Unitary CC")
print("-------------------")

# Energy of Eq. (23) 
def E_uCCD(t): 
 sigma = t * (JpJp_cc - JmJm_cc)
 expR = scipy.linalg.expm(sigma)
 expL = scipy.linalg.expm(-sigma)
 auxmat0 = np.matmul(expL, np.matmul(hamil, expR))
 energy = np.dot(mb_ref.transpose(), np.dot(auxmat0, mb_ref))
 
 return energy
 
 
#We solve uCCD by finding optimal t  
sol_uCCD = scipy.optimize.minimize(E_uCCD, 0., method='Nelder-Mead')
euccd = E_uCCD(sol_uCCD.x)
print(sol_uCCD.x, euccd) 



## End of unitary coupled cluster approach 

## Variational-generalized coupled cluster approach 

print("Variational-generalized CC")
print("-------------------")

# Energy of Eq. (28) 
def E_vgCCD(args):
 t, a, b = args[0], args[1], args[2] 
 cluster = t * JpJp_cc + a * JmJm_cc + b * JpJm_cc
 expR = scipy.linalg.expm(cluster)
 vgCCDvec = np.dot(expR, mb_ref)
 
 energy = np.dot(vgCCDvec.transpose(), np.dot(hamil, vgCCDvec))
 norm = np.dot(vgCCDvec.transpose(), vgCCDvec)
 
 energy = energy/norm 
 
 return energy
 
 
#We solve vgCCD by finding optimal t  
sol_vgCCD = scipy.optimize.minimize(E_vgCCD, [0.,1.,1.], method='Nelder-Mead')
evgccd = E_vgCCD(sol_vgCCD.x)
print(sol_vgCCD.x, evgccd) 



## End of variational-generalized coupled cluster approach 

print(" ")
print("Summary CC")
print("-------------------")
print("x, Exact, traditional, variational, unitary, variational-generalized")
print("-------------------")
print(x, min(wli).real, etccd, evccd, euccd, evgccd   )
if verbose == True:
 print(" ")
 print("Errors CC")
 print("x, traditional, variational, unitary, variational-generalized")
 print("-------------------")
 print(x, abs(etccd-min(wli))/N, abs(evccd-min(wli))/N, abs(euccd-min(wli))/N, abs(evgccd -min(wli))/N  )



