
flag: bool = True
while flag:
    try:
        list1 = ['10', '10', '100', '0', '0']
        print("list1 is ", list1)
        list2 = ['100', '0', '0', '10', '10']
        print('list2 is ', list2)
        listtemp = list()
        list1store = list()

        if list1.__len__() != list2.__len__():
            print(False)
            exit()
        n = list1.__len__()
        # multiplying the list one into 2 as the traversing can start from any point
        list1 *= 2
        # joining each element in the list1 and storing as string and will check if the string2 is in this string
        str1 = ''.join(list1)
        print('str1 is', str1)
        # joining each element in the list2 and storing as string
        str2 = ''.join(list2)
        print('str2 is ', str2)
        if str1.__contains__(str2):
            print(True)
        else:
            print(False)

    except IndexError:
        print('in index error ', list1store)
    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
