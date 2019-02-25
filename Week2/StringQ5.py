from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:

    try:
        list1 = []
        print("Enter the number of entries")
        num_input = util.get_positive_integer()
        for counter in range(0, num_input):
            list1.append(input("Enter the string"))
            # calling function to get the length of the longest string
        print("length of the longest string is ", util.longest_string(list1))
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
