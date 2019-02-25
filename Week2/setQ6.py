
flag: bool = True
while flag:
    try:
        s1 = {1, 2, 3, 14, 15}
        print(s1)
        s2 = {4, 5, 6, 1, 2}
        # storing the elements which are in both the sets
        s3 = s1 & s2
        print(s3)

    except Exception as exep:
        print("Process stopped because %s" % exep)
    print("To exit press 0 else press any other number")
    if input() == 0:
        flag = False
