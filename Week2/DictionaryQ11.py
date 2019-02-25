flag: bool = True
while flag:

    try:
        dict1 = {'1': 1}
        sets = [{'id': 1, 'success': True, 'name': 'Lary'}, {'id': 2, 'success': False, 'name': 'Rabi'}, {'id': 3, 'success': True, 'name': 'Alex'}]
        dict1.clear()
        count = 0
        for item in sets:
            # storing the index as the key and the values
            dict1[count] = item
            count += 1
        print(dict1)
    except Exception as e:
            print("Process stopped because %s" % e)

    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
