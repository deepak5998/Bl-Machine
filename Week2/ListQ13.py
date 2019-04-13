flag: bool = True
while flag:
    try:
        list1 = ['Red', 'yop', 'der', 'poy', 'op', 'edr']
        print("l1 is ", list1)
        list2 = ['pop', 'poy', 'Red']
        print('list2 is ', list2)
        for string in list2:
            # appending each word from list2 into list1
            list1.append(string)
        print("list1 is ", list1)

    except IndexError:
        print(list1)
    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
