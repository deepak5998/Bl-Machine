class HeadsAndTails:

    def __init__(self):
        self.num = 3
        self.list1 = ['HHH', 'HTH', 'HTT', "HHT", 'TTT', 'THT', 'THH', 'TTH']

    def probabilityOfHHH(self):
        # calculate the probability of HHH
        probability = self.list1.count('HHH')/len(self.list1)
        print("The probability of HHH when 3 coins tossed is {}/{} that is {}".format(self.list1.count('HHH'), len(self.list1), probability))

    def probabilityOf1H(self):
        # List having the elements having only 1 head
        list_1_H = [item for item in self.list1 if item.count('H') == 1]
        print("{} The probability of exactly 1 head is {}/{} that is {}".format(list_1_H, len(list_1_H), len(self.list1), len(list_1_H)/len(self.list1)))

    def atleast_2_Head_Given_1Head(self):
        # set of elements having atleast 1 head
        new_set = {item for item in self.list1 if item.count('H') > 0}
        # Set of the elements having the count of heads as 2 or more
        two_head_list = {item for item in self.list1 if item.count('H') > 1}
        print("{} \n The probability of atleast 2 heads is {}/{} that is {}".format(two_head_list, len(two_head_list), len(new_set), len(two_head_list)/len(new_set)))


try:
    head_tail_object = HeadsAndTails()
    flag: bool = True
    while flag:
        head_tail_object.probabilityOfHHH()
        head_tail_object.probabilityOf1H()
        head_tail_object.atleast_2_Head_Given_1Head()
        print("press 0 else press any other number")
        if input() == 0:
            flag = False
except Exception as e:
        print("Process stopped because %s" % e)

