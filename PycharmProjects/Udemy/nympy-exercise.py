import numpy as np

zeros = np.zeros(10)
print(zeros)

ones = np.ones(10)
print(ones)

fives = ones * 5
print(fives)

int_collection = np.arange(10, 51,1)
print(int_collection)

mat_pre = np.arange(0,9,1)
mat = mat_pre.reshape((3,3))
print(mat)

id_mat = np.eye(3)
print(id_mat)

rand_1 = np.random.rand(1)
print(rand_1)

norm_rand = np.random.randn(25)
print(norm_rand)

rand_interim = np.linspace(0, 1, num= 100)
rand_3 = rand_interim.reshape((10,10))
print(rand_3)

twenty_pts = np.linspace(0, 1, num= 20)
print(twenty_pts)


matrix  = np.arange(1,26, step = 1).reshape((5,5))
print(matrix)

print(matrix[2: , 1: ])
print(matrix[3, 4])
print(matrix[4, 0: ])
print(matrix[0:3, 1].reshape((3,1)))
print(matrix[4, 0 : ])
print(matrix.sum())
print(matrix.std())
sum_col = list()
for i in range(5) :
    currsum = 0
    for j in range(5):
        currsum = currsum + matrix[j,i]
    sum_col.append(currsum)
print(sum_col)