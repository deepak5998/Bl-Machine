flag: bool = True
while flag:
    try:
        set1 = {1, 2, 3, 14, 15}
        print(set1)
        set2 = {4, 5, 6, 1, 2}
        # creating the frozen set
        set3 = frozenset({1, 2, 3, 4})
        # getting type of the set
        print(type(set3))
        print(set3)
        set3 = set3 - set1
        print(set3)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

