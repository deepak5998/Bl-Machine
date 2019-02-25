flag: bool = True
while flag:

    try:
        string1 = "restart"
        string2 = string1[1:string1.__len__()]
        for counter1 in range(1, string1.__len__()):
                if string1[counter1] == string1[0]:
                    string1 = string1[0] + string2.replace(string1[counter1], '$')
        print(string1)
    except IndexError:
        print(string1)
    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
