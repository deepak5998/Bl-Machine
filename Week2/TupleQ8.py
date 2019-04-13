
flag: bool = True
while flag:
    try:
        tuple1 = tuple(['apple', 'dhoni', 'veejay', 'deepak', 'suhas'])
        print(tuple1)
        print("Enter the item to be removed")
        user_input = input()
        # as tuple is immutable storing the values of tuple in the list
        list1 = list(tuple1)
        # removing the element from the list
        try:
            list1.remove(user_input)
            print("successfully deleted")
        except:
            print("not found")
        # storing values back to the tuple from the updated list
        tuple1 = tuple(list1)
        print(tuple1)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
