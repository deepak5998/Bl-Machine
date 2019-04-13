
flag: bool = True
while flag:
    try:
        list1 = ['apple', 'dhoni', 'veejay', 'deepak', 'suhas']
        print(list1)
        # copying the list as tuple in new variable
        tuple1 = tuple(list1)
        print(tuple1)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
