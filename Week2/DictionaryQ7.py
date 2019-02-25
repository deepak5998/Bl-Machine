flag: bool = True
while flag:
    try:
        dict1 = {"IV": "S001", "VII": "S002", "VI": "S001", "IIV": "S005", "IIIV": "S005", "V": "S009", "VIII": "S007"}
        print(dict1)

        print("Printing the set created from sets")
        # storing as set as sets do not store duplicate elements
        llist = set(dict1.values())
        for counter in llist:
            print(counter, " ")
        print("To exit press 0 else press any other number")
        if input() == 0:
            flag = False

    except Exception as e:
        print("Process stopped because %s" % e)
