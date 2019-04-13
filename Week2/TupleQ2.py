flag: bool = True
while flag:
    try:
        # Storing different kind of values in the tuple
        tuple1 = tuple(['apple', 'banana', 1, 23, 54.343, 'f', [10, 20], {20}])
        print(tuple1)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

