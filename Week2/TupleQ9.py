flag: bool = True
while flag:
    try:
        tuple1 = tuple(['apple', 'dhoni', 'veejay', 'deepak', 'suhas'])
        print(tuple1)
        # slicing the tuple and storing in a new tuple
        tuple2 = tuple1[2:]
        print(tuple2)
        tuple3 = tuple1[:3]
        print(tuple3)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
