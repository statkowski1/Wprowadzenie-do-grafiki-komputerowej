import numpy as np

#zad1
x1 = np.linspace(5, 5, 50)
print(x1)

#zad2
x2 = np.arange(25).reshape(5, 5)
x2 = x2 + 1
print(x2)

#zad3
x3 = np.linspace(10, 50, 21)
print(x3)

#zad4
x4 = np.eye(5) * 8
print(x4)

#zad5
x5 = np.arange(100) * 0.01
x5 = x5.reshape(10, 10)
print(x5)

#zad6
x6 = np.linspace(0, 1, 50)
print(x6)

#zad7
x7 = x2[2:5,1:5]
print(x7)

#zad8
x8 = np.array([x2[0,-1], x2[1,-1], x2[2,-1]]).reshape(3, 1)
print(x8)

#zad9
rozmiar = x2.shape
x9 = x2[-1,:].sum() + x2[rozmiar[1]-2,:].sum()
print(x9)

#zad10
def losowy_tensor():
    tmp = np.random.randint(1, 10 ,(1, 2))
    return np.random.randint(1, 100, (tmp[0][0], tmp[0][1]))

print("Losowa macierz")
print(losowy_tensor())
