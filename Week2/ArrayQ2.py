from Utility.UtilityDataStructures import UtilityDataStructures
import array as ar
util = UtilityDataStructures()

flag: bool = True
while flag:

    try:
        print('Enter the number of elements to be added to the array')
        util = UtilityDataStructures()
        num = util.get_positive_integer()
        counter = 0
        # creating an empty array
        array = ar.array('i', [])
        print("Enter the elements for the array")
        # storing the elements by taking input from the user
        while counter in range(0, num):
            inputnum = util.get_integer()
            array.append(inputnum)
            counter += 1
        for counter in range(0, num):
            print(array[counter])
        # revering the array
        print("After using inbuilt reverse function")
        array.reverse()
        for counter in range(0, num):
            print(array[counter])

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
