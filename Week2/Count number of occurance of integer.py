from Utility.UtilityDataStructures import UtilityDataStructures
import array as array_object
flag: bool = True
while flag:
    try:
        print('Enter the number of elements to be added to the array')
        # object of utility class
        util = UtilityDataStructures()
        # input from user
        num = util.get_positive_integer()
        # empty array
        array = array_object.array('i', [])
        print("Enter the elements for the array")
        # counts number of times the number occured
        count = 0
        counter = 0
        # storing input from user
        while counter in range(0, num):
            numinput = util.get_integer()
            array.append(numinput)
            counter += 1

        print("Enter the number to be searched")
        # getting number to be searched
        search = util.get_integer()
        counter = 0
        for counter in range(0, num):
            if array[counter].__eq__(search):
                count += 1
        print("The number of occurrences  of ", search, " are ", count)
    except Exception as e:
        print("Process stopped because %s" % e)
    print("Enter 0 to exit another value to continue")
    if input() == 0:
        flag = False
