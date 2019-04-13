class Rain_Traffic_Late:

    def __init__(self):

        self.rain = 1 / 3
        self.no_rain = 2 / 3
        self.rain_traffic = 1 / 2
        self.rain_no_traffic = 1 / 2
        self.no_rain_traffic = 1 / 4
        self.no_rain_no_traffic = 3 / 4
        self.rain_traffic_late = 1 / 2
        self.no_rain_no_traffic_late = 1 / 8
        self.rain_no_traffic_late = self.no_rain_traffic_late = 1/4
        self.no_rain_traffic_not_late = 1 - self.no_rain_traffic_late

    def not_Raining_Traffic_not_Late(self):
        probability = self.no_rain * self.no_rain_traffic * self.no_rain_traffic_not_late
        return probability

    def late(self):
        p_rain_traffic_late = self.rain * self.rain_traffic * self.rain_traffic_late
        p_rain_no_traffic_late = self.rain * self.rain_no_traffic * self.rain_no_traffic_late
        p_no_rain_traffic_late = self.no_rain * self.no_rain_traffic * self.no_rain_traffic_late
        p_no_rain_no_traffic__late = self.no_rain * self.no_rain_no_traffic * self.no_rain_no_traffic_late
        probability = p_rain_traffic_late + p_rain_no_traffic_late + p_no_rain_traffic_late + p_no_rain_no_traffic__late
        return probability

    def given_Late_Probaility_Of_Rain(self):
        p_late_rain_traffic = self.rain * self.rain_traffic * self.rain_traffic_late
        p_late_rain_no_traffic = self.rain * self.rain_no_traffic * self.rain_no_traffic_late
        probability = p_late_rain_traffic + p_late_rain_no_traffic
        return probability/self.late()


try:
    rain_object = Rain_Traffic_Late()
    flag: bool = True
    while flag:
        print("It's not raining and there is heavy traffic and I am not late "
              , rain_object.not_Raining_Traffic_not_Late()
              , "\nProbability that I am late ", rain_object.late()
              , "\nI arrived late at work probability that it rained that day is "
              , rain_object.given_Late_Probaility_Of_Rain()
              , "\nTo exit press 0 else press any other number")
        if input() == 0:
                    flag = False
except Exception as e:
        print("Process stopped because %s" % e)

