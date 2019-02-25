flag: bool = True
while flag:
    try:
        # creating the new set
        set1 = {1, 2, 3}
        print(set1)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

