class Bernouli:

    # class static variables
    success = 1
    fail = 0
    p_success_error = 0.1
    p_fail_no_error = 1 - p_success_error
    mean = 0
    variance = 0
    population = 1000

    @staticmethod
    def calculating_Mean():
        Bernouli.mean = Bernouli.success * Bernouli.p_success_error + Bernouli.fail * Bernouli.p_fail_no_error
        print("Mean is ", Bernouli.mean)

    @staticmethod
    def calculating_Variance():
        Bernouli.variance = Bernouli.p_success_error * (1 - Bernouli.p_success_error)

        print("The variance is ", Bernouli.variance)

    @staticmethod
    def calculating_Standard_Deviation():
        # standard deviation is square root of variance
        Bernouli.standard_dev = Bernouli.variance ** 0.5
        print("The Standard deviation is ", Bernouli.standard_dev)

    @staticmethod
    def calculating_Z_score(error_x):
        # z score = x - N*mean / standard_dev * sqrt(N)
        Bernouli.z_score_x = (error_x - (Bernouli.population * Bernouli.mean)) /\
                             (Bernouli.standard_dev * (Bernouli.population ** 0.5))
        return Bernouli.z_score_x

    def calculating_probaility(self):
        Bernouli.calculating_Mean()
        Bernouli.calculating_Variance()
        Bernouli.calculating_Standard_Deviation()
        # converting x to precision 2
        x = "%.2f" % Bernouli.calculating_Z_score(120)
        print("The Zscore is ", x)
        probability_error = 0.5832
        return 1 - probability_error


try:
    bernouli_object = Bernouli()
    flag: bool = True
    while flag:
        print("The probability of error 120 is {:f}".
              format(bernouli_object.calculating_probaility()), "\nTo exit press 0 else press any other number")
        if input() == 0:
                    flag = False
except Exception as e:
        print("Process stopped because %s" % e)
