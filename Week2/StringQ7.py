flag: bool = True
while flag:

    try:
        string1 = (input("Enter the string"))
        print(string1)
        stringlist = list(string1.split(','))
        slist = []
        for counter in range(0, stringlist.__len__()):
            stringlist[counter] = stringlist[counter].replace(' ', '')
        stringlist.sort()
        for counter1 in range(stringlist.__len__()):
                if slist.__contains__(stringlist[counter1]):
                    continue
                else:
                    slist.append(stringlist[counter1])
        print(slist)

    except Exception as e:
        print("Process stopped because %s" % e)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
