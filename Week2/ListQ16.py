flag: bool = True
while flag:
    try:
        l1 = ['Red', 'yop', 'der', 'poy', 'op', 'edr']
        print("l1 is ", l1)
        # creating the list of just the first character in each item of the list
        l2 = [item[0] for item in l1]
        print(l2)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
