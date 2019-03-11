# _*_ coding: utf-8 _*_

"""
All the entities in th project.

Author: Genpeng Xu
Date:   2019/03/07
"""


class PredictionResult:
    """
    The accuracy results of prediction, where `m_acc` represents the accuracy of M month,
    `m1_acc` represents the accuracy of M+1 month, and so on.
    """

    def __init__(self, m_acc, m1_acc, m2_acc, m3_acc):
        self._m_acc = m_acc
        self._m1_acc = m1_acc
        self._m2_acc = m2_acc
        self._m3_acc = m3_acc

    @property
    def m_acc(self):
        return self._m_acc

    @m_acc.setter
    def m_acc(self, m_acc):
        self._m_acc = m_acc

    @property
    def m1_acc(self):
        return self._m1_acc

    @m1_acc.setter
    def m1_acc(self, m1_acc):
        self._m1_acc = m1_acc

    @property
    def m2_acc(self):
        return self._m2_acc

    @m2_acc.setter
    def m2_acc(self, m2_acc):
        self._m2_acc = m2_acc

    @property
    def m3_acc(self):
        return self._m3_acc

    @m3_acc.setter
    def m3_acc(self, m3_acc):
        self._m3_acc = m3_acc

    def __str__(self):
        return "Accuracy: M = %.2f%%  M+1 = %.2f%%  M+2 = %.2f%%  M+3 = %.2f%%" % (self._m_acc * 100,
                                                                                   self._m1_acc * 100,
                                                                                   self._m2_acc * 100,
                                                                                   self._m3_acc * 100)


if __name__ == '__main__':
    pr = PredictionResult(0.78, 0.65, 0., 0.)
    print(pr)
