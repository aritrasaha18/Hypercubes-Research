import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

# Finding the shortest paths in Hypercube graphs using harmonic extension
# @author B D S Aritra & Ryan H. Pellico
# @version 1.0 07/19/2022

# Builds the adjacency matrix and the binary strings
# Takes in the dimension of the cube
# returns the adjacency matrix with binary strings for vertices
def cube_VA(n):
  # initialize A and V with adjacency matrix and bit strings for n = 1 case
  A = np.array([[0,1],[1,0]])
  V = ['0','1']

  count = 1
  # if n>2, recursively build next V,A from previous V,A
  while count < n:
      I = np.eye(2**count,dtype = int)
      A1 = np.concatenate((A,I),axis = 1)
      A2 = np.concatenate((I,A),axis = 1)
      A = np.concatenate((A1,A2),axis = 0)
      V0 = V.copy()
      V1 = V.copy()
      for i in range(0,len(V1)):
          V0[i] = '0' + V0[i]
          V1[i] = '1' + V1[i]
      V = V0 + V1
      count+=1
  return V, A

# Builds the phi values of all the vertices
# Takes in the value of the boundary points and the adjacency Matrix
# returns the phi values
def phi_func(A,r1,r2):
  # A is the binary symmetric adjacency matrix of a connected simple graph G
  # r1 and r2 are the row indices (0,1,2,..) of u and w, resp.
  # where u and w are the boundary vertices where g(u) = 1, g(w) = -1
  # output phi is the unique harmonic extension of g to the graph G,
  # represented as a column vector

  # g1 and g2 are the boundary data (values of g at u and w)
  g1 = 1
  g2 = -1
  #p1 = np.array([1],dtype = float)
  #p2 = np.array([-1], dtype = float)

  # (Diagonal) Degree Matrix
  D = np.diag(np.sum(A,axis = 1))
  # (Combinatorial Graph) Laplacian Matrix
  L = np.subtract(D,A)

  # Set up the linear system for phi, L'*phi = b, which has a symmetric, diagonally
  # dominant (SDD) matrix L', obtained by deleting from L*phi = 0 the two rows
  # corresponding to the boundary vertices, and absorbing appropriate constants
  # from the columns corresponding to the boundary vertices into the column
  # vector b on the right-hand side of the linear system
  L = np.delete(L,[r1,r2],axis = 0)
  b = np.add(np.multiply(-g1,L[:, r1]),np.multiply(-g2,L[:, r2]))
  L = np.delete(L,[r1,r2],axis = 1)

  # Solve the linear system for phi
  phi = np.linalg.solve(L, b)

  # since we only solved for the *unknown* values of phi, we must put the known
  # values of phi(u) = 1 and phi(w) = -1 into the approriate positions, r1 and r2
  if (r1 < r2):
    phi = np.insert(phi,r1,g1)
    phi = np.insert(phi,r2,g2)
  else:
    phi = np.insert(phi,r2,g2)
    phi = np.insert(phi,r1,g1)

  # round in the 12th decimal place so that numerical error does not affect two
  # equal numbers seeming different, like 1e-18, 0 and 2e-20.
  # Claim: in theory phi values could differ by less than any epsilon>0 while
  # still not being equal, but not for graphs with "few" vertices
  phi = np.round(phi,12)

  return phi

# Finds all the paths in a graph
# Takes in the adjacency matrix and the boundary points
# returns the paths
def find_all_path(A,u,w):
    # first solve for phi
    phi = phi_func(A,u,w)
    # PATHS will be a list of all "phi-paths" which start at u where g = +1,
    # and always proceed to the neighbor where phi is at a minimum, which
    # forces eventual arrival at w where g = -1 and the path terminates.
    PATHS = [[u]]
    # currs will contain the list of all terminal vertices of the paths in PATHS
    currs = [];
    for i in range(len(PATHS)):
        currs.append(PATHS[i][-1])
    # if any of the paths in PATHS end in w, the loop will end since a path was found
    while (currs.count(w) == 0):
      NEW_PATHS = [] # will contain one or more "extended paths" for each path in PATHS
      # loop over each path in PATHS
      for i in range(len(PATHS)):
        path = PATHS[i] # the current path
        curr = path[-1] # the (currently) last vertex in 'path'
        N = np.argwhere(A[curr,:]>0) # the indices of the neighbors of 'curr'
        PHI = phi[N] # the 'phi' values at the neighbors of 'curr'
        next = N[PHI == min(PHI)] # the indices of the neighbor(s) of 'curr' where 'phi' is smallest
        # loop over all indices in 'next', and create a new extended path (from 'path') for each
        for j in range(len(next)):
          new_path = path.copy()
          new_path.append(next[j])
          NEW_PATHS.append(new_path)
      # reset variables 'PATHS' and 'currs' for next iteration
      PATHS = NEW_PATHS
      currs = []
      for i in range(len(PATHS)):
        currs.append(PATHS[i][-1])
    return PATHS

# Visual Representation of the Hypercube
# Takes in the adjacency matrix, x coordinates, y coordinates, phi values & binary strings
# returns the graph of the Hypercube
def draw_cube(A,x,y,phi,V):
    dx = 0.03
    dy = 0.03
    for i in range(0,len(A)):
        t = plt.text(x[i]-dx,y[i]+dy,'Φ('+V[i]+') = ' + str(Fraction(phi[i]).limit_denominator()),verticalalignment='top',fontsize=12,color = 'red',fontstyle = 'oblique',ha = 'right',va = 'bottom',bbox=dict(boxstyle="round",ec=(0.1, 0.1, 0.1),fc=(1., 0.5, 0.8)))
        t.set_bbox(dict(facecolor='pink', alpha=0.5, edgecolor='pink'))
        for j in range(i,len(A)):
            if(A[i][j]==1):
                plt.plot([x[i],x[j]],[y[i],y[j]],'-b',marker = 'o',mec = 'r',markersize = 10,linewidth = 2)

    plt.scatter(x,y)
    plt.axis('off')
    plt.grid(b=None)
    plt.show()

print("******************************Hypercube Generator*******************************")
print("")
n = int(input("Enter the dimension of the cube: "))
u = int(input("Boundary point 1: "))
w = int(input("Boundary point 2: "))
x = np.random.random_sample((2**n,))
y = np.random.random_sample((2**n,))
V,A = cube_VA(n)
phi = phi_func(A,u,w)
PATHS = find_all_path(A,u,w)
print("Number of paths from ",V[u]," ( u =",u,") to ",V[w]," ( w =",w,") is ",len(PATHS))
print(PATHS)
draw_cube(A,x,y,phi,V)
