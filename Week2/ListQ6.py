
from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:

    try:
        list1 = [2, 5, 1, 2, 4, 4, 2, 3, 2, 1]
        list2 = []
        print(list1)
        # iterating the list
        for element in list1:
            if list2.__contains__(element):
                continue
            else:
                # storing the elements in the list which have not yet encountered
                list2.append(element)
        print("Using  diff list ", list2)
    except Exception as e:
        print("Process stopped because %s" % e)
    '''
        for i in range(list1.__len__()-1 ):
            for j in range(i+1, list1.__len__()-1):
                if list1[i] == list1[j]:
                    list1 = util.swipe_from_r_to_l(list1, j, list1.__len__())
                    list1.remove(list1[j])
        print("singly ",list1)

    except IndexError:
        print("singly ",list1)
    '''

    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
