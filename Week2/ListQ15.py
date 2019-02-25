flag: bool = True
while flag:
    try:
        list1 = ['Red', 'yop', 'der', 'poy', 'op', 'edr']
        print("l1 is ", list1)
        list2 = ['pop', 'poy', 'Red']
        print('l2 is ', list2)
        list3 = []
        for string in list1:
            # storing just the common elements in the list1 and list2
            if list2.__contains__(string):
                list3.append(string)
        print("Common items in l1 and l2 are ", list3)
    except IndexError:
        print(list1)
    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
