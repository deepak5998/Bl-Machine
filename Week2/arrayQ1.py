'''
Write a Python program to create an array of 5 integers and display the array items. Access individual element through indexes.
'''
import array as array_object
flag: bool = True
while flag:

    try:
        array1 = array_object.array('i', [1, 2, 3, 4])
        # printing elements in the array
        for counter in array1:
            print(counter)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
