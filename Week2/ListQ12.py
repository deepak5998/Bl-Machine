flag: bool = True
while flag:

    try:
        list1 = ['Red', 'yop', 'der', 'poy', 'op', 'edr']
        print("list11 is ", list1)
        list2 = ['pop', 'poy', 'Red']
        print('list2 is ', list2)
        list3 = []
        for string1 in list1:
            if list2.__contains__(string1):
                continue
            else:
                # storing the elements which are in list1 but not in list2
                list3.append(string1)

        print('l3 is ', list3)
    except IndexError:
        # if after sorting the index occurs it enters here
        print(list1)
    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
