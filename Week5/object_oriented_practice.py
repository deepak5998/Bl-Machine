from Utility.UtilityDataStructures import UtilityDataStructures
import datetime
import pickle


class Object_Oriented_Problems:
    util = UtilityDataStructures()

    def __init__(self):
         print()

    def read_from_file(self, file_name):
        file = open(file_name, 'r')
        data = file.readline()
        return data

    def is_Date(self, date):
        try:
            datetime.datetime.strptime(date, '%d-%m-%Y')
        except ValueError:
            raise ValueError("Incorrect data, should be DD-MM-YY")

    def take_input_from_user(self, main_data):
        data = []
        print('Enter the name')
        while True:
            name = input()
            flag = False
            if (Object_Oriented_Problems.util.is_string(name)):
                for item in main_data:
                    if item[0] == name:
                        flag = True
                        print("Name already present try again")
                if not flag:
                    break
        data.append(name)
        print('Enter the date ob birth')
        while True:
            date_obj = input()
            if (Object_Oriented_Problems.util.is_Date(date_obj)):
                break
        data.append(str(date_obj))
        print('Is date secret?\nif yes enter yes else no')
        while True:
            status = input().lower()
            if status == 'yes':
                break
            if status =='no':
                break
            else:
                print('please enter yes or no')

        data.append(status)
        print("Successfully entered data")
        return data

    def execute(self):
        data = []
        try:
            file = open('objectoriented.txt', 'w')
        except Exception as e:
            print("could not open file")
        try:
            while True:
                print('Enter 1 to enter data\n2 to check in data\n3to exit')
                option = Object_Oriented_Problems.util.get_positive_integer()
                if option == 3:
                    try:
                        file.write(str(data))
                        # pickle.dump(data, 'objectoriented.txt')
                        print("written data successfully")
                    except Exception as e:
                        print("Could no write data as ",e)
                    exit()
                if option == 1:
                    data.append(self.take_input_from_user(data))
                elif option == 2:
                    if len(data) == 0:
                        print("There is no data to search")
                    else:
                        flag = False
                        print("Enter the name to search")
                        while True:
                            name = input()
                            if Object_Oriented_Problems.util.is_string(name):
                                break
                        for item in data:
                            if item[0] == name:
                                flag = True
                                print('data found')
                                if item[2]=='no':
                                    print('Date of birth :', item[1])
                                else:
                                    print("date of birth is secret")
                        if not flag:
                            print('data not found')
        except Exception as e:
            print("The process stopped as ", e)


object_oriented = Object_Oriented_Problems()
object_oriented.execute()
