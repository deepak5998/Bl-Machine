flag: bool = True
while flag:

    try:
        list1 = ['abc', 'xyz', 'aba', '1221']
        print(list1)
        # creating a new list of those elements whose length is more than 1 and the first and last element is the same
        list2 = [item for item in list1 if item[0] == item[-1] and item.__len__() > 1]
        print('number of elements is ', list2.__len__(), ' and the values are ', list2)
        print("To exit press 0 else press any other number")
        if input() == 0:
            flag = False
    except Exception as e:
        print("Process stopped because %s" % e)


