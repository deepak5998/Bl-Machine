from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:

    try:
        list1 = ['Prayas', 'Deepak', 'Sunil', 'Veejay', 'Suhas', 'Narayan', 'Ajay']
        print("Enter the length ")
        length = util.get_positive_integer()
        # iterating through the list
        for string in list1:
            # printing only those whose length is more than length specified by user
            if string.__len__() > length:
                print(string)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
