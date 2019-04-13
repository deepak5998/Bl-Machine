flag: bool = True
while flag:
    try:
        string1 = input("Enter a string")
        # checking if the length is more than 3
        if string1.__len__() < 3:
            print("no change")
        else:
            if string1.endswith("ing"):
                # if already has ing
                string1 += 'ly'
            else:
                string1 += 'ing'
            print(string1)
    except IndexError:
        print(string1)
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
