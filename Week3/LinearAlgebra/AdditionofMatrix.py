flag: bool = True
while flag:
    try:
        matrix1 = [[12, 7, 3], [4, 5, 6], [7, 8, 9]]
        print(matrix1)
        matrix2 = [[5, 8, 1], [6, 7, 3], [4, 5, 9]]
        print(matrix2)
        # adding the elements at the same positions
        matrix3 = [[matrix1[row][col] + matrix2[row][col] for row in range(0, 3)] for col in range(0, 3)]
        print(matrix3)
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
