flag: bool = True
while flag:
    try:
        matrix1 = [1, 2, 3]
        matrix2 = [[5, 1, 3], [1, 1, 1], [1, 2, 1]]
        item = 0
        if matrix1.__len__() != matrix2.__len__():
            print("can't multiply")
            exit()
        matrix3 = [0 for col in range(0, len(matrix1))]
        for row in range(0, matrix2.__len__()):
            item = 0
            # Vector multiplication
            for col in range(0, len(matrix2)):
                item += matrix1[col] * matrix2[col][row]
            matrix3[row] = item
    except Exception as e:
        print("Process stopped because %s" % e)
    print("matrix 1 {} \nmatrix2 is {} \nmultiplication is {}".format(matrix1, matrix2, matrix3)
          , "\nTo exit press 0 else press any other number")
    if input() == 0:
        flag = False
