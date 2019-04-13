flag: bool = True
while flag:
    try:
        tuple1 = ('apple', 1.23, 54.343, 'f', [10, 20], {20}, 'apple')
        tuple2 = []
        count = 0
        for counter in range(tuple1.__len__()):
            count = 0
            for counter2 in range(counter + 1, tuple1.__len__()):
                # tuple 2 will store  the values if it is duplicated in the tuple1 and  was not in tuple2
                if tuple1[counter] == tuple1[counter2] and (tuple2.__contains__(tuple1[counter2])) is False:
                    tuple2.append(tuple1[counter2])
        print(" the elements having the duplicates are ", tuple2)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False

