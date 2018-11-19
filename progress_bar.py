import sys

class ProgressBar():

    def __init__(self, maxValue, prefix='', suffix='', decimals=2, barLength=100):
        self.maxValue = maxValue
        self.curValue = 0
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.barLength = barLength

    def printProgress(self, value):
        """
        Call in a loop to create terminal progress bar

        Parameters::

            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : number of decimals in percent complete (Int)
            barLength   - Optional  : character length of bar (Int)

        Based on: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        """

        filledLength = int(round(self.barLength * value / float(self.maxValue)))
        percents = 100.00 * (value / float(self.maxValue))
        bar = '#' * filledLength + '-' * (self.barLength - filledLength)
        sys.stdout.write('%s [%s] %*.*f%s %s\r' % (self.prefix, bar, self.decimals+3, self.decimals, percents, '%', self.suffix))
        sys.stdout.write('\n')
        sys.stdout.flush()

    def update(self, value):
        self.curValue = value
        self.printProgress(value)

    def increment(self, value):
        self.update(self.curValue + value)
