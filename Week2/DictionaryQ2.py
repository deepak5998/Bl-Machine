from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:
    try:
        dict1 = {"0": 11, "1": 13, "2": 12, "3": 16, '4': 14}
        print("The dictionary before sorting is")
        # printing the dictionary
        for key in dict1:
            print("For key ", key, " value is: ", dict1[key])
        # taking user input
        print("Enter the key to be added")
        inputkey = util.get_integer()
        print("Enter the value to be added")
        inputvalue = util.get_integer()

        print("the dict after manipulation with key")
        # adding the input to the dictionary
        dict1[inputkey.__str__()] = inputvalue
        print(dict1)
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
