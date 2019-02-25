from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:

    try:
        list1 = [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
        list2 = []
        print(list1)
        # iterate through each element in the list
        for counter in range(list1.__len__()):
            # for the items after the current element iterated by the outer loop
            for counter2 in range(counter + 1, list1.__len__()):
                if util.equalsList(list1[counter], list1[counter2]):
                    list1.remove(list1[counter2])
        print(list1)
    except IndexError:
        print(list1)
    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
