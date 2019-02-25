from copy import deepcopy
flag: bool = True
while flag:
    try:
        tuple1 = tuple(['apple', 1.23, 54.343, 'f', [10, 20], {20}])
        # copying the tuple to different tuple
        tuple2 = tuple1
        # or using deepcopy
        tuple2 = deepcopy(tuple1)
        print(tuple1)
        print(tuple2)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
