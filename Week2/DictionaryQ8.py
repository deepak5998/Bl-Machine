flag: bool = True
while flag:

    try:
        # creating an empty dictionary
        dict1 = dict()
        string = 'w3resource'
        dict1.clear()
        # storing the values in the dictionary along with their count
        for counter in string:
            dict1[counter] = string.count(counter)
        print(dict1)
        print("To exit press 0 else press any other number")
        if input() == 0:
            flag = False
    except Exception as e:
        print("Process stopped because %s" % e)
