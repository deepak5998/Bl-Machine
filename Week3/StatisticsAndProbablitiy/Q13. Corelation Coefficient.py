class Correlation_Coefficient:

    def __init__(self):
        self.set_x = list([68, 72, 65, 70, 62, 75, 78, 64, 68])
        self.set_y = list([90, 85, 88, 100, 105, 98, 70, 65, 72])
        self.mean_x = self.mean_y = 0
        self.standard_dev_x = self.standard_dev_y = 0
        self.correlation_coefficient = 0
        # learing to make generators
        # self.generator_object_x = self.generator(self.set_x)
        # self.generator_object_y = self.generator(self.set_y)

    # generator
    def generator(self, list_object):
        for num in list_object:
            yield num

    def calculate_mean_x_and_y(self):
        for item in self.set_x:
            self.mean_x += item
        self.mean_x /= len(self.set_x)
        for item in self.set_y:
            self.mean_y += item
        self.mean_y /= len(self.set_y)

    def calculate_standard_deviation_x_and_y(self):
        # self.standard_dev_x += ((item - self.mean_x) ** 2 for item in self.set_x)
        # self.standard_dev_y += ((item - self.mean_y) ** 2 for item in self.set_y)
        for item in self.set_x:
            self.standard_dev_x += (item - self.mean_x) ** 2
        self.standard_dev_x /= (len(self.set_x) - 1)
        self.standard_dev_y **= 1 / 2
        for item in self.set_y  :
            self.standard_dev_y += (item - self.mean_y) ** 2
        self.standard_dev_y /= (len(self.set_y) - 1)
        self.standard_dev_y **= 1 / 2

    def z_score_calculator_x(self, point_x):
        # calculating Z score for point x using mean and standard deviation for that object
        return (point_x - self.mean_x) / self.standard_dev_x

    def z_score_calculator_y(self, point_y):
        # calculating Z score for point x using mean and standard deviation for that object
        return (point_y - self.mean_y) / self.standard_dev_y

    def calculate_Correlation_Coefficient(self):
        # self.correlation_coefficient += (self.z_score_calculator_x(item_x) * self.z_score_calculator_y(item_y)
        #                                 for item_x, item_y in zip(self.set_x, self.set_y))
        # calculating correlation co-efficient
        for item in range(0, len(self.set_x)):
            x_part = (self.set_x[item] - self.mean_x) / self.standard_dev_x
            y_part = (self.set_y[item] - self.mean_y) / self.standard_dev_y
            self.correlation_coefficient += x_part * y_part
        self.correlation_coefficient /= (len(self.set_x)-1)

    def print_All(self):
        print(
            "xmean is {:f}, y mean is {:f}\n"
            "x standard deviation is {:f}, y standard deviation is {:f}\n"
            "correlation coefficient is {:f}"
                .format(correlation_coeff_object.mean_x, correlation_coeff_object.mean_y,
                        correlation_coeff_object.standard_dev_x,
                        correlation_coeff_object.standard_dev_y, correlation_coeff_object.correlation_coefficient)
        )


try:
    correlation_coeff_object = Correlation_Coefficient()
    flag: bool = True
    while flag:

        correlation_coeff_object.calculate_mean_x_and_y()
        correlation_coeff_object.calculate_standard_deviation_x_and_y()
        correlation_coeff_object.calculate_Correlation_Coefficient()
        correlation_coeff_object.print_All()
        print("To exit press 0 else press any other number")
        if input() == 0:
                flag = False
except Exception as e:
        print("Process stopped because %s" % e)
