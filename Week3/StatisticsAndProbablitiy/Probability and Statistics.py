from Utility.UtilityDataStructures import UtilityDataStructures


class Cards:

    def __init__(self):
        self.cards = 52
        self.kings = 4
        self.ace = 4

    def ace_In_Pack_Of_Cards(self):
        print("Probability of drawing an ace from cards is {}/{} that is {}".format(self.ace, self.cards, self.ace / self.cards))

    def ace_After_King(self):
        self = Cards()
        self.cards -= 1
        prob_ace = self.ace / self.cards
        print("The probability of getting an Ace after a king is taken is {}/{} that is {}".format(self.ace, self.cards, prob_ace))

    def ace_After_Ace(self):
        self.ace -= 1
        self.cards -= 1
        print("The probability od getting an ace after first getting an ace is {}/{} that is {}".format(self.ace, self.cards,self.ace / self.cards))


try:
    card_object = Cards()
    util = UtilityDataStructures()
    string_list = ("Ace independent", "Ace after King", "Ace after Ace")
    num_input = 1
    while num_input != 0:
        for counter in range(0, len(string_list)):
            print("for {} press {}".format(string_list[counter], counter + 1))
        print("and press 0 to exit")
        num_input = util.get_positive_integer()
        if num_input == 1:
            card_object = Cards()
            card_object.ace_In_Pack_Of_Cards()
        elif num_input == 2:
            card_object = Cards()
            card_object.ace_After_King()
        elif num_input == 3:
            card_object = Cards()
            card_object.ace_After_Ace()
        elif num_input == 0:
            exit()
        else:
            print("Not yet created")
except Exception as e:
                print("Process stopped because %s" % e)
