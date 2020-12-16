import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def eleven():
    A = np.array([(0, 2, 4),(2, 4, 2), (3, 3, 1)])
    b = np.array([(-2),(-2),(-4)])
    c = np.array([(1),(1),(1)])

    AInv = np.linalg.inv(A)
    print("Part A")
    print("Inverse, A^(-1): ", AInv)

    print("Part B")
    print("A Inverse @ b: ", AInv@b)

    print("Matrix Multiplication, (A@c): ", A@c)

def twelve():
    n = 40000
    Z=np.random.randn(n)
    plt.step(sorted(Z), np.arange(1,n+1)/float(n))

    ks = [1, 8, 64, 512]
    for k in ks:
        Z=np.sum(np.sign(np.random.randn(n, k))*np.sqrt(1./k), axis=1)
        plt.step(sorted(Z), np.arange(1,n+1)/float(n))
    plt.legend(["Gaussian"]+ks)
    plt.xlim((-3,3))
    plt.show()
    
if __name__ == "__main__":
    eleven()
    twelve()
