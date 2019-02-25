flag: bool = True
while flag:
    try:
        set1 = {1, 2, 3, 14, 15}
        print(set1)
        set2 = {4, 5, 6, 1, 2}
        print(set2)
        # storing the elements which are either of the sets
        set3 = set1 | set2
        print(set3)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input == 0:
        flag = False

