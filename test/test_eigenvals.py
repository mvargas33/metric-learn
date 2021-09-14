import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eigh, eig

# Rotate the function used to calculate the eigenvalues
def _eigh(a, b, dim, i=0):
  if i%3 == 0:
    try:
      return eigsh(a, k=dim, M=b, which='LA')
    except np.linalg.LinAlgError:
      pass  # scipy already tried eigh for us
    except (ValueError, ArpackNoConvergence):
      pass
  elif i%3 == 1:
    try:
      return eigh(a, b)
    except np.linalg.LinAlgError:
      pass
  else:
    return eig(a, b)

def find_error_eigenvalues():
  attempt = 1
  while True:
    a = np.random.rand(2,2)
    a = (a + a.T) / 2 # Symemtrice
    
    prev = []
    n = set()
    for i in range(10000):
      vals, vecs = _eigh(a, np.identity(np.shape(a)[0]), np.shape(a)[0], i)
      for v in vecs:
        if v[0] not in prev: # First value of an eigenvector
          prev.append(v[0])
          _ = [n.add(x) for x in vals]

    if len(set(prev)) % np.shape(a)[0] > 0:
      print(f'Found {len(n)} eigenvalues {n}')
      print(f'First {len(set(prev))} values of vectors found {prev}')
      print(f'Aattempts: {attempt}')
      return
    attempt += 1

find_error_eigenvalues()