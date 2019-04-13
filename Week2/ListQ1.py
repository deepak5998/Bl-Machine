flag: bool = True
while flag:

    try:
        # initialising the list
        list1 = [1, 2, 3, 4, 5]
        print(list1)
        # variable to store the addition of the values in the list
        total = 0
        for elements in list1:
            # adding the values of the list
            total += elements
        print(total)
        print("To exit press 0 else press any other number")
        if input() == 0:
            flag = False

        print("also by sum method", sum(list1))
    except Exception as e:
        print("Process stopped because %s" % e)
