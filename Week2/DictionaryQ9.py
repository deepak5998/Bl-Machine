flag: bool = True
while flag:
    try:
        dict1 = {'1': 1}
        string = 'w3resource'
        dict1.clear()
        for counter in string:
            dict1[counter] = string.count(counter)
            # printing the key and values as tables
        for (key, vals) in dict1.items():
            print(key, " \t", vals)
    except Exception as e:
            print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
