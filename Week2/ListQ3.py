
from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:

    try:
        list1 = [1, 2, 3, 4, -1, 5]
        # printing the min value in the list using the min function
        print("Smallest element is ", min(list1))
        print("To exit press 0 else press any other number")
        if input() == 0:
            flag = False
    except Exception as e:
        print("Process stopped because %s" % e)
