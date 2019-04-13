
flag: bool = True
while flag:
    try:
        set1 = {1, 2, 3, 14, 15}
        print(set1)
        if 16 in set1:
            # element is in the list then only it'l be removed
            set1.remove(16)
            print("successfully deleted")
        else:
            # if the element is not present there it'll show as not present
            print("Not present in the set")
        print(set1)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

