flag: bool = True
while flag:

    try:
        list1 = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
        print(list1)
        # sorting the variables according to the second element in each tuple of list
        for counter1 in range(0, list1.__len__() - 1):
            for counter2 in range(counter1 + 1, list1.__len__()):
                if list1[counter1][1] > list1[counter2][1]:
                    temp = list1[counter1]
                    list1[counter1] = list1[counter2]
                    list1[counter2] = temp

        print(list1)
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
