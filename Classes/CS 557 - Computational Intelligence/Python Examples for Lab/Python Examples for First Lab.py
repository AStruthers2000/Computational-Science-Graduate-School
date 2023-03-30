import numpy as np

def randomarray(n,m):
  b = [[np.random.randint(1,11) for i in range(n)] for j in range(m)]
  return b       
        
        
a = np.array([[2,2,2,2],[2,2,2,2], [2,2,2,2], [2,2,2,2], [2,2,2,2], [2,2,2,2]])
print (a) 

print("\nSum:")
# In Python 2 this is print "Sum:")
print(a+a)      



c = [[0 for i in range(6)] for j in range(4)]
c = randomarray(6,4)
print("\nTranspose")
print(np.transpose(c))



c = [[0 for i in range(4)] for j in range(4)]
c = randomarray(4,4)

print("\nInverse:")
print(np.linalg.inv(c))
print("\nDeterminant:")
print(np.linalg.det(c))
print("\nEigenvalues and Eigenvectors:")
print(np.linalg.eig(c))


import matplotlib.pyplot as plt

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
plt.plot(t, s)
plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('Function')
plt.grid(True)
plt.savefig("test.png")
plt.show()
