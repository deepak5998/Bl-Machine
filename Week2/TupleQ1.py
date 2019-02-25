flag: bool = True
while flag:
    try:
        # creating a tuple
        tuple1 = tuple({'apple', 'banana'})
        print(tuple1)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

