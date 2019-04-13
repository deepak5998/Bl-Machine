import itertools
flag: bool = True
while flag:

    try:
        # initializing the list
        list1 = ['Red', 'yop', 'der', 'poy', 'op', 'edr']
        # converting all the elements in lowercase
        for counter in range(0, list1.__len__()):
            list1[counter] = list1[counter].lower()
        print(list1)
        # this list will store the characters returned by function permutation
        charperms = []
        # it stores the words when combined
        listperms = []
        num = list1.__len__()
        for item in range(0, list1.__len__()):
            # emptying the list for new use
            charperms = ['']
            # storing the values returned by permuting the each word in the list
            charperms = itertools.permutations(list1[item])
            listperms.clear()
            for each in charperms:
                # storing all the permutations of each word
                listperms.append(''.join(each))
            for others in list1[item + 1:]:
                if listperms.__contains__(others):
                    list1.remove(others)
        for counter in range(0, list1.__len__()):
            print(list1[counter])
    except IndexError:
        print(list1)
    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
