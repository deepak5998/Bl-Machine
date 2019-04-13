flag: bool = True
while flag:
    try:
        # set of dictionaries
        dict1 = {'': ''}
        dict1.clear()
        sets = [{'id': 1, 'success': True, 'name': 'Lary'}, {'id': 2, 'success': False, 'name': 'Rabi'}, {'id': 3, 'success': True, 'name': 'Alex'}]
        {dict1.update(item) for item in sets}
        for item in dict1.items():
            print(item)
        count = 0
        # taking from sets as the dictionary does not have multiple values for same keys
        for item in sets:
            if item['success']:
                # the dictionary has the value True at key success in it
                count += 1
        print("The success has the value True ", count, " times")

    except Exception as e:
        # printing the exception
            print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
