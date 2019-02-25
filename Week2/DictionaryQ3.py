
flag: bool = True
while flag:
    try:
        # empty dictionary
        dictnew = {'': ''}
        dict1 = {0: 11, 1: 13}
        dict2 = {2: 12, 3: 16}
        dict3 = {4: 14, 5: 17}
        print("The dictionaries before storing are")
        print("The 3 dictionaries are")
        print(dict1)
        print(dict2)
        print(dict3)

        # clearing the dictionary for storing the new values
        dictnew.clear()

        # copy dict1 to new dictionary
        dictnew = dict1.copy()

        # updating the dictionary if keys not present then gets added
        dictnew.update(dict2)
        dictnew.update(dict3)

        print("the resultant dict is")
        print(dictnew)

        for key in dictnew:
            print("key: ", key, " value: ", dictnew[key])
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
