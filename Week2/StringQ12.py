from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:

    try:
        string1 = "APPLEBANANA"
        print(string1)
        print("Enter the value of n")
        # taking a number input from user
        num_input = util.get_positive_integer()
        if not num_input > string1.__len__():
            # storing the 2 parts of string lowering first part and rest part as it is
            string1 = string1[:num_input].lower() + string1[num_input:]
            print(string1)
        else:
            print("The number is more than the size of the string")
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
