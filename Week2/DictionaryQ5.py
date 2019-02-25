from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:

    try:
        dict1 = {'': ''}
        dict1.clear()
        print("Enter the number")
        inputnum = util.get_positive_integer() + 1
        # storing the values in the dictionary
        dict1 = {(counter, counter * counter) for counter in range(1, inputnum)}
        print(dict1)
        print("To exit press 0 else press any other number")
        if input() == 0:
            flag = False
    except Exception as e:
        print("Process stopped because %s" % e)

