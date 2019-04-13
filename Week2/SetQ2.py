flag: bool = True
while flag:
    try:
        # creating a new  list
        set1 = {1, 2, 3}
        # iterating through the elements in the set
        for item in set1:
            print(item)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

