flag: bool = True
while flag:
    try:
        tuple1 = ('apple', 1.23, 54.343, 'f')
        # unpacking the values in different variables
        tuple0, tuple2, tuple3, tuple4 = tuple1
        # printing each of the variables taken from tuple
        print(tuple0)
        print(tuple2)
        print(tuple3)
        print(tuple4)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

