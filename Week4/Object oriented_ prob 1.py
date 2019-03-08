''' Write a class which takes input from console name of the person, and his/her date of birth.
Then when the person’ name is typed it should display the dob.
b) Some persons dob are secret so those should not be displayed, but will be stored.
If a secret person name is entered then you should just display “secret”.
However all the data should be persisted, and loaded. We can add to the list.
What data structure would you create for these 2 problems.
'''
import pandas as pd
from Utility.UtilityDataStructures import UtilityDataStructures
import datetime as dt

class Birth_Date:

    def __init__(self):
        self.name = list()
        self.dob = list()
        self.util = UtilityDataStructures()

    def sayhi(self):
        print('hi')

    def take_input(self):
        # print("Enter the name")
        # string = input()
        # while(not self.util.is_string(string)):
        #     string = input()
        # self.name.append(string)
        # print(self.name)
        # print('Enter date of birth')
        # try:
        #     dob = dt(input())
        # except Exception:
        #     print("Not a proper input please try again")
        # print(dob)

        print("Enter a number")
        num = self.util.get_positive_integer()
        print(num)


birth_date = Birth_Date()
birth_date.take_input()
