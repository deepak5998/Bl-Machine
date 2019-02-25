class Normal_Distributed:

    def __init__(self):
        self.mean = 30
        self.standard_deviation = 4

    def __init__(self, mean=None, standard_deviation=None):
        # if mean and standard deviation is not provided
        if not mean and not standard_deviation:
            self.mean = 30
            self.standard_deviation = 4
        else:
            self.mean = mean
            self.standard_deviation = standard_deviation

    def z_score_calculator(self, point_x):
        # calculating Z score for point x using mean and standard deviation for that object
        return (point_x - self.mean) / self.standard_deviation

    def probability_x_less_than_40(self, point_x):
        print("z_score = ", self.z_score_calculator(point_x))
        value_z_score = 0.9938
        return value_z_score

    def probability_x_more_than_21(self, point_x):
        print("z_score = ", self.z_score_calculator(point_x))
        value_z_score = 0.0122
        # as the prob of finding val to the right of 21 will be 1 - prob of finding to left of 21
        return 1 - value_z_score

    def probability_between_30_and_35(self, point_x1, point_x2):
        z_score_30 = self.z_score_calculator(point_x1)
        print("z_score for 30 = ", z_score_30)
        z_score_35 = self.z_score_calculator(point_x2)
        print("z_score for 35 = ", z_score_35)
        value_z_score_30 = 0.5
        value_z_score_35 = 0.8944
        return value_z_score_35 - value_z_score_30

    # Q.11
    def probability_For_Radar_Unit(self, point_x):
        print("In the radar Unit object z_score = ", self.z_score_calculator(point_x))
        value_z_score = 0.8413
        return 1 - value_z_score


try:
    normal_distributed_object = Normal_Distributed()
    flag: bool = True
    while flag:
        radar_object = Normal_Distributed(90, 10)
        print("The probability of area < 40 is ", normal_distributed_object.probability_x_less_than_40(40)
              , "\nThe probability of area more than 21 is ", normal_distributed_object.probability_x_more_than_21(21)
              , "\nThe probability of area between 30 and 35 is "
              , normal_distributed_object.probability_between_30_and_35(30, 35)
              , "\nThe probability that a car picked at random is travelling at more than 100 km/hr is %.4f" %
              radar_object.probability_For_Radar_Unit(100)
              , "\nTo exit press 0 else press any other number")
        if input() == 0:
                    flag = False
except Exception as e:
        print("Process stopped because %s" % e)

