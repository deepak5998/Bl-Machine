from Utility.UtilityDataStructures import UtilityDataStructures
util = UtilityDataStructures()
flag: bool = True
while flag:

    try:
        list1 = ['Prayas', 'Deepak', 'Sunil', 'Veejay', 'Suhas', 'Narayan', 'Ajay']
        list2 = ['Rakesh', 'sunil', 'Manoj']
        print(list1)
        print(list2)
        # call the function and it the value returned is true prints common else says no common element
        if util.if_common_element(list1, list2):
            print("the list have common element")
        else:
            print("no common element")

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
