flag: bool = True
while flag:
    try:
        set1 = {1, 2, 3, 14, 15}
        print(set1)
        # clearing the set
        set1.clear()
        print(set1)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

