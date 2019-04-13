flag: bool = True
while flag:
    try:
        set1 = {1, 2, 3, 4, 5, 5, 6, 7, -1, -3}
        print(set1)
        # printing the min value in the set
        print(min(set1))
        # printing the max value in the set
        print(max(set1))

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
