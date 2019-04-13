
flag: bool = True
while flag:
    try:
        tuple1 = tuple(['apple', 'dhoni', 'veejay', 'deepak', 'suhas'])
        print(tuple1)
        # reversing the tuple and storing the output as a tuple in the variable
        tuple2 = tuple(reversed(tuple1))
        print(tuple2)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
