flag: bool = True
while flag:
    try:
        list1 = [2, 5, 1, 2, 4, 4, 2, 3, 2, 1]
        list2 = []
        # cloning the list 1 and storing it in the list1
        list2 = list1.copy()
        print(list1)
        print(list2)
    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
