#Computes coupled matrix elements of a general quadrupole plus monopole (J=0) pairing interaction
import sys
import numpy as np
import itertools as it
from scipy.special import gamma, factorial, assoc_laguerre 
from scipy.integrate import quad


#Input values for the interaction 

output_name      = "QQQ"          #Interaction output filename
conf_space       = ["103","101"]  #Configuration space. Shells in Antoine's format
shells_eners     = [0.,0.]        #Energies of the former shells (MeV)
PP_coef, QQ_coef = 0., 1.         #Scaling of the J=0 pairing and quadrupole interaction: H_total = PP_coeff H_PP + QQ_coeff H_QQ
ho_b             = 1.00132        #Harmonic oscillator constant b (fm)

 
assert len(shells_eners) == len(conf_space) 

### Routines needed 

#Clebsch-Gordan coefficients. Adapted from HFODD 10.1088/1361-6471/ac0a82
def cg(J1,M1,J2,M2,J,M): #YOU HAVE TO ENTER 2*J1, 2*M1... ALL INTEGERS 
    
  assert (isinstance(J1, int) == True)
  assert (isinstance(M1, int) == True)
  assert (isinstance(J2, int) == True)
  assert (isinstance(M2, int) == True)
  assert (isinstance(J, int) == True)
  assert (isinstance(M, int) == True)

  MV25=150
  LSIZ=MV25

  H, L = np.zeros(MV25,dtype=float), np.zeros(MV25,dtype=int)
  H[0] = 1.0
  L[0] = 0
  LAST = 1
  SCALE1 = 8.0 

  cgcoef=0.0
  
  if (J < 0):
   print('J<0')
   return cgcoef

  if (J1 < 0): 
   print('J1<0')
   return cgcoef      

  if (J2 < 0):
   print('J2<0')
   return cgcoef

  if (abs(M) > J):
   print('abs(M) > J')
   return cgcoef

  if (abs(M1) > J1):
   print('abs(M1) > J1')
   return cgcoef

  if (abs(M2) > J2):
   print('abs(M2) > J2')
   return cgcoef                   

  if (M1 + M2 != M):
   print('M1+M2 != M')
   return cgcoef

  if (np.mod(J1+J2+J,2) != 0):
   print('J1+J2+J NOT EVEN')
   return cgcoef


  I0=int((J1+J2+J)/2+1)

  if (J1+J2-J < 0):
   print('J1+J2-J < 0')
   return cgcoef

  I1=int((J1+J2-J)/2)

  if (J < abs(J1-J2)):
   print('J < |J1-J2|')
   return cgcoef   
      
  I2=int((J-(J1-J2))/2)
  I3=int((J+(J1-J2))/2)

  if (np.mod(J -M ,2) != 0):  
   print('MOD(J-M,2) != 0')
   return cgcoef                    
  
  I8=int((J +M )/2)
  I9=int((J -M )/2)


  if (np.mod(J1-M1,2) != 0):
   print('MOD(J1-M1,2) != 0')
   return cgcoef
                       
  I4=int((J1+M1)/2)
  I5=int((J1-M1)/2)
      
  if (np.mod(J2-M2,2) != 0):
   print('MOD(J2-M2,2) != 0')
   return cgcoef

  I6=int((J2+M2)/2)
  I7=int((J2-M2)/2)
  N2=J2-J-M1
  N3=J1-J+M2
  N4=min(I1,I5,I6)
  N5=max(0,N2,N3)
  N1=int(N5/2)
    

  if (N1 > N4):
   print('N1 > N4')
   return cgcoef                 
      
  MM1=max(I0,I1,I2,I3,I4,I5,I6,I7,I8,I9,N4-int(N2/2),N4-int(N3/2))+1

  if (MM1 <= LAST):

    IY =  (L[I1]+L[I2]+L[I3]+L[I4]+L[I5]+L[I6]+L[I7]+L[I8]+L[I9]-L[I0])/2

    Y=np.sqrt(H[I1]*H[I2]*H[I3]*H[I4]*H[I5]*H[I6]*H[I7]*H[I8]*H[I9]/H[I0]*(J+1))
      
        
    if (np.mod(N5,4) != 0): Y=-Y

    Z=0.0

    for N5 in range(N1,N4):
     MM1=I1-N5
     MM2=I5-N5
     MM3=I6-N5
     MM4=N5-N2/2
     MM5=N5-N3/2
      
     X=Y/H[MM1]/H[MM2]/H[MM3]/H[MM4]/H[MM5]/H[N5]
     IX= L[MM1]+L[MM2]+L[MM3]+L[MM4]+L[MM5]+L[N5]

     if (IY-IX < 0):
      X=X/SCALE1
      IX=IX-1
     if (IY-IX > 0):
      X=X*SCALE1
      IX=IX+1
     if (IY-IX == 0):
      Z=Z+X

    Y=-Y
    cgcoef=Z
    return cgcoef
    

  if (MM1 >= LSIZ):
    print('ERROR: larger factorials are needed')
    print('Increase the size MV25 of arrays H and L in routine cg')
    sys.exit()

 
  X=LAST
  MM2=LAST+1
  LAST=MM1
  for MM3 in range(MM2,MM1+1):
    H[MM3-1]=H[MM3-2]*X
    L[MM3-1]=L[MM3-2]
    if ( H[MM3-1] < SCALE1 ):
      X=X+1.0 
      continue 
    while ( H[MM3-1] > SCALE1 ):
      H[MM3-1]=H[MM3-1]/SCALE1**2
      L[MM3-1]=L[MM3-1]+2
      if ( H[MM3-1] < SCALE1 ):
       X=X+1.0 
       continue 

  IY =  int((L[I1]+L[I2]+L[I3]+L[I4]+L[I5]+L[I6]+L[I7]+L[I8]+L[I9]-L[I0])/2)

  Y=np.sqrt(H[I1]*H[I2]*H[I3]*H[I4]*H[I5]*H[I6]*H[I7]*H[I8]*H[I9]/H[I0]*(J+1))
    
  if (np.mod(N5,4) != 0): Y=-Y
  
  Z=0.0

  for N5 in range(N1,N4+1):
   MM1=I1-N5
   MM2=I5-N5
   MM3=I6-N5
   MM4=N5-int(N2/2)
   MM5=N5-int(N3/2)
      
   X=Y/H[MM1]/H[MM2]/H[MM3]/H[MM4]/H[MM5]/H[N5]
   IX= L[MM1]+L[MM2]+L[MM3]+L[MM4]+L[MM5]+L[N5]

   while (IY-IX < 0):
    X=X/SCALE1
    IX=IX-1
   while (IY-IX > 0):
    X=X*SCALE1
    IX=IX+1
   if (IY-IX == 0):
    Z=Z+X

   Y=-Y

  cgcoef=Z
    
  return cgcoef       

def krond(a,b):   #Kronecker delta
    krond = 0
    if (a == b):
      krond = 1
    return krond


# 3j Symbols:  / J1 J2 J3 \
#              \ M1 M2 M3 /

def threej(J1,M1,J2,M2,J,M): #YOU HAVE TO ENTER 2*J1, 2*M1... ALL INTEGERS 

 c = cg(J1,M1,J2,M2,J,-M)
 jh  = np.sqrt(J+1.)
 return (-1)**(.5*(J1-J2-M))/jh*c
 

# 6j Symbols:  / J1 J2 J12 \
#              \ J3 J0 J23 /
# Formula (1.59) Suhonen, applying the condition on the 3j of M1+M2+M3=0 for being nonzero
#YOU HAVE TO ENTER 2*J1, 2*M1... ALL INTEGERS 
def sixj(J1,J2,J12,J3,J0,J23):

 sj = 0.
 for m1 in range(-J1,J1+1,2):
  for m2 in range(-J2,J2+1,2):
   for m3 in range(-J3,J3+1,2):
    m12 = -(m1+m2)
    m23 = m2+m3
    m0  = m1+m2+m3
    
    if (abs(m12) > J12): continue
    if (abs(m23) > J23): continue
    if (abs(m0) > J0): continue        
    
    phas = (-1)**round(.5*(J3+J0+J23-m3-m0-m23))
    tj12 = threej(J1,m1,J2,m2,J12,m12) 
    tj10 = threej(J1,m1,J0,-m0,J23,m23) 
    tj32 = threej(J3,m3,J2,m2,J23,-m23) 
    tj30 = threej(J3,-m3,J0,m0,J12,m12)             

    sj += phas*tj12*tj10*tj32*tj30

  
 return sj
 
  
#Harmonic oscillator wf according to Suhonen (3.42) 
def gnl(x,n,l,b=1.): 

 gfac = np.sqrt(2.*factorial(n)/b**3/gamma(n+l+1.5))
 pol = assoc_laguerre(x**2/b**2,n,l+0.5)
 
 gnlx = gfac * (x/b)**l * np.exp(-x**2/2./b**2) * pol

 return gnlx
 
def norm_gnl(x,l,na,la,nb,lb,b=1.): 

 return x**(l+2)*gnl(x,na,la,b)*gnl(x,nb,lb,b)

 
#Radial integral matrix elements according to Suhonen (6.41). General shell input
def redradQsh(a,b,l):
 #l=lambda for the multipole. l=2 is the quadrupole 
 
 shells = [a,b]
 
 nv, lv, jv = [], [], []
 
 for sh in shells:

  if (len(sh) == 3):
   nsh = 0                               #n values
   lsh = int(sh[0])                       #l values 
   jsh = .5*float(sh[1:3])                #j values
  else: 
   nsh = int(sh[0])                   #n values
   lsh = int(sh[1])                   #l values
   jsh = .5*float(sh[2:4])              #j values
  
  nv.append(nsh)
  lv.append(lsh)
  jv.append(jsh) 
  
 na,nb = nv[0],nv[1]
 la,lb = lv[0],lv[1]
 ja,jb = jv[0],jv[1] 
 
 if (np.mod(la+lb+l,2) != 0):
  rab = quad(norm_gnl, 0, 50, args=(l,na,la,nb,lb))
  assert np.isclose(rab[1], 0.)
  return rab[0]


 ta, tb = round(.5*(lb-la+l)), round(.5*(la-lb+l))
 if (ta < 0): ta = 0
 if (tb < 0): tb = 0
 ga, gb = gamma(na+la+1.5), gamma(nb+lb+1.5)
 rfac = (-1)**(na+nb)*np.sqrt(factorial(na)*factorial(nb)/ga/gb)*factorial(ta)*factorial(tb)
 
 smin, smax = max(0,na-ta,nb-tb), min(na,nb)
 rs=0.
 for s in range(smin,smax+1):
  rs += gamma(.5*(la+lb+l)+s+1.5)/factorial(s)/factorial(na-s)/factorial(s+ta-na)/factorial(s+tb-nb)
  
 rab = rfac * rs
  
 return rab  

 
#Quadrupole reduced matrix elements according to Suhonen (6.23). General shell input
def redQ2sh(a,b,bho=1.):
 #i_a,i_b are the indices of HO levels
 #bgi is the HO constant
 #For quadrupole, l=lambda=2
 l=2

 shells = [a,b]
 
 nv, lv, jv = [], [], []
 
 for sh in shells:

  if (len(sh) == 3):
   nsh = 0                               #n values
   lsh = int(sh[0])                       #l values 
   jsh = .5*float(sh[1:3])                #j values
  else: 
   nsh = int(sh[0])                   #n values
   lsh = int(sh[1])                   #l values
   jsh = .5*float(sh[2:4])              #j values
  
  nv.append(nsh)
  lv.append(lsh)
  jv.append(jsh) 
  
 na,nb = nv[0],nv[1]
 la,lb = lv[0],lv[1]
 ja,jb = jv[0],jv[1]  
 
 fac = 1./np.sqrt(4.*np.pi)*(-1)**round(jb-l-.5)*.5*(1.+(-1)**round(la+lb+l))
 lh,jah,jbh = np.sqrt(2.*l+1.), np.sqrt(2.*ja+1.), np.sqrt(2.*jb+1.)
 
 if (ja+jb-l<0.): return 0.
 tj = threej(round(2*ja),round(2*1/2),round(2*jb),-round(2*1/2),2*l,2*0)
 

 rab = bho**l * redradQsh(a,b,l)
 qab = fac*lh*jah*jbh*tj*rab

 return qab 

#Formula 8.55 (8.67) from Suhonen !!!  Make the derivation for interest...
def Q2mubb(a,b,c,d,J,T,bho=1.):

 #if (np.mod(J+T,2) == 0 ): return 0.
 dab, dcd = 0., 0.
 if (a == b): dab = 1.
 if (c == d): dcd = 1.
 nab, ncd = np.sqrt(1.-dab*(-1)**round(J+T))/(1.+dab), np.sqrt(1.-dcd*(-1)**round(J+T))/(1.+dcd)
 
 shells = [a,b,c,d]
 
 nv, lv, jv = [], [], []
 
 for sh in shells:

  if (len(sh) == 3):
   nsh = 0                               #n values
   lsh = int(sh[0])                       #l values 
   jsh = .5*float(sh[1:3])                #j values
  else: 
   nsh = int(sh[0])                   #n values
   lsh = int(sh[1])                   #l values
   jsh = .5*float(sh[2:4])              #j values
  
  nv.append(nsh)
  lv.append(lsh)
  jv.append(jsh) 

 ja,jb,jc,jd = jv[0],jv[1],jv[2],jv[3]  
 
 #Symmetries of 6j symbols 
 if (ja+jb-J<0.): return 0.
 if ( (ja+jc-2.<0.) or (jb+jd-2.<0.) or (jc+jd-J<0.)): 
  Aabcd = 0.
 else: 
  Aabcd = (-1)**round(ja+jb+J)* sixj(round(2*ja),round(2*jb),2*J,round(2*jd),round(2*jc),2*2)*redQ2sh(c,a,bho)*redQ2sh(b,d,bho)

 
 if ( (ja+jd-2.<0.) or (jb+jc-2.<0.) or (jc+jd-J<0.) ): 
  Aabdc = 0.
 else:
  Aabdc = (-1)**round(ja+jb+J)* sixj(round(2*ja),round(2*jb),2*J,round(2*jc),round(2*jd),2*2)*redQ2sh(d,a,bho)*redQ2sh(b,c,bho) 
 

 VabcdJ = nab * ncd * (Aabcd + (-1)**round(jc+jd+J+T)*Aabdc)
 
 return  -2*VabcdJ  ### WHY -2 HERE??? To obtain the same as antoine...

#Formula 12.11 from Suhonen. J,T can not be anything else... 
def pair_int(a,b,c,d,J=0,T=1):
 dab, dcd = 0., 0.
 if (a == b): dab = 1.
 if (c == d): dcd = 1.
 
 shells = [a,b,c,d]
 
 nv, lv, jv = [], [], []
 
 for sh in shells:

  if (len(sh) == 3):
   nsh = 0                               #n values
   lsh = int(sh[0])                       #l values 
   jsh = .5*float(sh[1:3])                #j values
  else: 
   nsh = int(sh[0])                   #n values
   lsh = int(sh[1])                   #l values
   jsh = .5*float(sh[2:4])              #j values
  
  nv.append(nsh)
  lv.append(lsh)
  jv.append(jsh) 

 ja,jb,jc,jd = jv[0],jv[1],jv[2],jv[3]  
 la,lb,lc,ld = lv[0],lv[1],lv[2],lv[3] 
 
 Vp = -.5*(-1)**round(la+lc)*np.sqrt(2.*ja+1.)*np.sqrt(2.*jc+1)*dab*dcd
 
 return Vp
 
 
def write_to_file(filename,shells,sp_eners,c_pp,c_qq,bho):

 ns = len(shells)
 assert ns == len(sp_eners)

 
 combs = list(it.combinations_with_replacement(shells, 2))
 perms = []
 for i in range(len(combs)):
  for j in range(i,len(combs)):
   perms.append([combs[i][0], combs[i][1], combs[j][0], combs[j][1]])

  
 ff = open(filename,'w')
 ff.write('coupled PP + (or) QQ\n') #Title
 ff.write("   " + str(1) + "    " + str(len(shells)) + "     ")
 for i in shells:     #Write shell information
  ff.write(str(i) + "  ")
 ff.write('\n')
 ff.write("   ")
 for i in sp_eners:   #Write single particle energies
   ff.write(str(i) + "  ")
 ff.write('\n')
 ff.write("      0      0     0.0   1.00000000\n")  #Mass dependence
 for p in range(len(perms)):
  a, b, c, d = perms[p][0], perms[p][1], perms[p][2], perms[p][3]

     
  ja, jb, jc, jd = int(str(a)[-1]), int(str(b)[-1]), int(str(c)[-1]), int(str(d)[-1])
  jab_min, jab_max = .5*abs(ja-jb), .5*(ja+jb)
  jcd_min, jcd_max = .5*abs(jc-jd), .5*(jc+jd)     
     
  j0, j1 = int(jab_min), int(jab_max)
  if (jab_min < jcd_min): j0 = int(jcd_min)
  if (jab_max > jcd_max): j1 = int(jcd_max)
    
  if (j1 < j0): continue   
  ff.write("   0     1   " + str(a) +" "+ str(b)+" "+ str(c)+" "+ str(d)+" "+ str(j0)+ " "+ str(j1) + "\n")     
  
  for t in [0,1]:
   for j in range(j0,j1+1):
    Vmel = 0.
    if (abs(c_qq) > 0.):
     Vmel = c_qq * Q2mubb(a,b,c,d,j,t,bho)
    if (j==0 and t==1): Vmel += c_pp * pair_int(a,b,c,d) 
    Vmelstr = "{:.4f}".format(Vmel)
    ff.write(" " + Vmelstr + " ")
   ff.write("\n")
       
     
 
 ff.close() 
 
## START   

# Write to file the coupled matrix elements of QQ and PP interactions 
write_to_file(output_name, conf_space, shells_eners, PP_coef, QQ_coef, ho_b) 
 


