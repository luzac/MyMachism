from numpy import*

randMat = mat(array([[1, 2, 3, 4],
                     [1, 2, 3, 4],
                     [1, 2, 3, 4],
                     [1, 2, 3, 4]]))

# randMat = mat(random.rand(4, 4))
IRandMat = randMat.I
print(IRandMat * randMat - eye(4))
