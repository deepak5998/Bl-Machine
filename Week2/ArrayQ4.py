from Utility.UtilityDataStructures import UtilityDataStructures
import array as array_object
flag: bool = True
while flag:
    try:
        print('Enter the number of elements to be added to the array')
        util = UtilityDataStructures()
        num = util.get_positive_integer()
        counter = 0
        array = array_object.array('i', [])

        print("Enter the elements for the array")
        while counter in range(0, num):
            inputnum = util.get_integer()
            array.append(inputnum)
            counter += 1

        # printing the array
        for counter in range(0, num):
            print(array[counter], " ")

        print("Enter the number to search and delete")
        search = util.get_integer()
        try:
            # removing the element in the array
            array.remove(search)
            num -= 1
            print("After deleting the element in the array")

            # print the members of the array
            for counter2 in range(0, num):
                print(array[counter2], " ")
        except:
            print("Number not found in the array")

    except Exception as e:
        print("Process stopped because %s" % e)
    print("Enter 0 to exit another value to continue")
    if input() == 0:
            flag = False
