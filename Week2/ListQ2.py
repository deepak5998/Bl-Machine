flag: bool = True
while flag:

    try:
        list1 = [1, 2, 3, 4, 5]
        print(list1)
        # variable to store the multiplication of the values in the list
        total = 1
        for item in list1:
            # multiplying the values in the list
            total *= item
        print(total)
        print("To exit press 0 else press any other number")
        if input() == 0:
            flag = False
    except Exception as exep:
        print("Process stopped because %s" % exep)
