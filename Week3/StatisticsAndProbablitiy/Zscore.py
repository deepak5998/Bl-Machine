class Bank_Teller:

    def __init__(self):
        self.mean = 2
        self.variance = 1
        self.standard_deviation = self.variance ** 0.5

    def z_Score(self, n, point_x):
        # zscore = X- Mean / standard deviation * sqrt(n)
        return (point_x - self.mean * n) / (self.standard_deviation * (n ** (1/2)))

    def probability_Calculator_Z_Score(self, sample_size, point_1, point_2):
        # calculate z score for point 1
        z_score_1 = self.z_Score(sample_size, point_1)
        # calculate z score for point 2
        z_score_2 = self.z_Score(sample_size, point_2)
        print("Z score 1 is {} \nZ score 2 is {}".format(z_score_1, z_score_2))
        # reading values from the Z score table
        val_for_z_1 = 0.00749
        val_for_z_2 = 0.9251
        return val_for_z_2 - val_for_z_1


try:
    bank_teller_object = Bank_Teller()
    flag: bool = True
    while flag:
        probability = bank_teller_object.probability_Calculator_Z_Score(50, 90, 110)
        print("probability is ", probability, "\nTo exit press 0 else press any other number")
        if input() == 0:
                    flag = False
except Exception as e:
        print("Process stopped because %s" % e)
