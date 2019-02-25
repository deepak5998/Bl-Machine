flag: bool = True
while flag:

    try:
        dict1 = {"0": 11, "1": 13, "2": 12, "3": 16, '4': 14}
        print("The dictionary before sorting is")
        for key in dict1:
            print("For key ", key, " value is: ", dict1[key])

        keys = [key for key in dict1]
        values = [dict1[key] for key in dict1]
        temp = 0
        outer_counter = 0
        # sorting dictionary in  ascending order
        for outer_counter in range(0, values.__len__()):
            # the loop for all the elements other than the current one
            for inner_counter in range(outer_counter + 1, values.__len__() - 1):
                if values[inner_counter] > values[inner_counter + 1]:
                    temp = keys[inner_counter]
                    keys[inner_counter] = keys[inner_counter + 1]
                    keys[inner_counter + 1] = temp
                    temp = values[inner_counter]
                    values[inner_counter] = values[inner_counter + 1]
                    values[inner_counter + 1] = temp
        # store empty values again in dictionary to make it clear
        dict1 = {'': ''}
        print("The dictionary after sorting is")
        # storing values in dictionary after sorting
        for inner_counter in range(keys.__len__()):
            dict1[keys[inner_counter]] = values[inner_counter]
        dict1.__delitem__('')
        print("Now printing dictionary")
        print(dict1)

        print("In descending order")
        # sorting in descending order
        for outer_counter in range(values.__len__()):
            for inner_counter in range(values.__len__() - 1):
                if values[inner_counter] < values[inner_counter + 1]:
                    temp = keys[inner_counter]
                    keys[inner_counter] = keys[inner_counter + 1]
                    keys[inner_counter + 1] = temp
                    temp = values[inner_counter]
                    values[inner_counter] = values[inner_counter + 1]
                    values[inner_counter + 1] = temp
        dict1 = {'': ''}
        for inner_counter in range(keys.__len__()):
            dict1[keys[inner_counter]] = values[inner_counter]
        dict1.__delitem__('')
        print("Now printing dictionary")
        print(dict1)

    except Exception as e:
        print("Process stopped because %s" % e)
        '''
        print("Sorting with the help of items")
        for j in range(dict.__len__()):
            i=0
            for (k,v)  in dict.items():
                if i < (dict.__len__() - 1):
                    if dict[k[i]] > dict[k[i + 1]]:
                        item2 = dict[k[i]]
                        dict[k[i]] = dict[k[i + 1]]
                        dict[k[i + 1]] = item2
                else: break
                i +=1
        print("Now printing dictionary")
        
        print(dict)
    '''
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
