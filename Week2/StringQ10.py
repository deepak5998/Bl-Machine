flag: bool = True
while flag:

    try:
        string1 = "learnappleappleap"
        # counting the number of occurrences of a string
        print(string1.count('le'))
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
