class Patients:

    def __init__(self):
        self.p_given_pain_kiler = 0.1
        self.p_addicted = 0.05
        self.p_addicted_given_painkiller = 0.08
        self.p_painkiler_and_addicted = 0

    def calculate_Painkiller_And_Addicted(self):
        # P(A and B) = P(B/A) * P(A)
        self.p_painkiler_and_addicted = self.p_addicted_given_painkiller * self.p_given_pain_kiler

    def calculate_Painkiller_Given_Addicted(self):
        self.calculate_Painkiller_And_Addicted()
        # P(A/B) = P(A and B) / P (B)
        return self.p_painkiler_and_addicted / self.p_addicted


try:
    patient = Patients()
    flag: bool = True
    while flag:
        print("The probability of patient being addicted is ", patient.calculate_Painkiller_Given_Addicted(),
              "\nTo exit press 0 else press any other number")
        if input() == 0:
                    flag = False
except Exception as e:
        print("Process stopped because %s" % e)

