flag: bool = True
while flag:
    try:
        set1 = {1, 2, 3}
        print(set1)
        # adding the new element in the set
        set1.add(12)
        set1.add(13)
        print(set1)
    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

