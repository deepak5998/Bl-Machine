flag: bool = True
while flag:
    try:
        sets = [{'id': 1, 'success': True, 'name': 'Lary'}, {'id': 2, 'status': False, 'naam': 'Rabi'}, {'id': 3, 'success': True, 'name': 'Alex'}]
        flag = True
        keylist = list(sets[0].keys())
        count = len(keylist)
        print(keylist)
        # all the elements in the set
        for element in sets:
            # for all the keys and values in the dictionary
            for key, val in element.items():
                # restoring the value of the flag to true
                flag = True
                # so that it is executed untill all the elements are iterated
                for ke in keylist:

                    if keylist.__contains__(key):
                        flag = False
                        break
                if not flag:
                    # it means that the element is present
                    continue
                else:
                    # if not present add the key to the key list
                    keylist.append(key)
                    count += 1
        print(keylist, "and number of different keys is ", count)
        print("To exit press 0 else press any other number")
        if input() == 0:
            flag = False
    except Exception as e:
            print("Process stopped because %s" % e)
