flag: bool = True
while flag:
    try:
        matrix1 = [[1, 2, 3], [3, 4, 5], [7, 6, 4]]
        matrix2 = [[5, 2, 6], [5, 6, 7], [7, 6, 4]]
        print("Matrix1 is ", matrix1, "\nMatrix2 is ", matrix2)
        matrix3 = [[0 for row in range(3)] for col in range(3)]

        for row in range(0, matrix1[0].__len__()):
            for col in range(0, len(matrix2)):
                for count in range(0, matrix2[0].__len__()):
                    # each element is multiplication matrix is
                    # multiplication of row elements of matrix 1 and column elements of matrix 2
                    matrix3[row][col] += matrix1[row][count] * matrix2[count][col]
    except Exception as e:
        print("Process stopped because %s" % e)
    print("The multiplication of the matrices is", matrix3
          , "\nTo exit press 0 else press any other number")
    if input() == 0:
        flag = False
