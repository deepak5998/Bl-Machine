from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:
    try:
        set1 = {1, 2, 3, 12, 13, 14}
        print(set1)
        # remove the element from the list
        set1.remove(12)
        print(set1)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

