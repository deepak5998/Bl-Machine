flag: bool = True
while flag:

    try:
        string1 = "https://www.w3resource.com/python-exercises"
        print(string1)
        char_input = input("Enter the character")
        if not string1.__contains__(char_input):
            print("Character not in string")
        else:
            stringout = ""
            for item in string1:
                if item == char_input:
                    break
                stringout += item
            print(stringout)
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
