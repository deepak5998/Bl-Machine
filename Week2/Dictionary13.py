flag: bool = True

while flag:
    try:
        # creating a new dictionary
        dict1 = {'1': 1}
        # clearing the dictionary for use
        dict1.clear()
        # creating set of dictionary
        sets = [{'id1': '1', 'successs': True, 'names': 'Lary'},
                {'id': '2', 'status': False, 'naam': 'Rabi'},
                {'id': '3', 'success': True, 'name': 'Alex'}]
        # adding elements to the dictionary
        for io in sets:
            dict1.update(io)
        print(dict1)
        val = []
        count = 0
        # for key, value in dictionary items
        for (key, value) in dict1.items():
            # storing unique vals in new list
            if val.__contains__(value):
                continue
            else:
                val.append(value)
                count += 1
        print(val, " and count is ", count)

    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
            flag = False
