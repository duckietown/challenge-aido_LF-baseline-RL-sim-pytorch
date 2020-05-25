from math import floor

class Conv2d():
    def __init__(self, input, output, kernel, padding=0, dilation=1, stride=1):
        self.kernel = kernel
        self.output=output
        self.padding=padding
        self.dilation=dilation
        self.stride=stride

    def __call__(self, input):
        def c(x):
            y = x+2*self.padding-self.dilation*(self.kernel-1)-1
            y /= self.stride
            return floor(y+1)

        ret = self.output, c(input[1]), c(input[2])
        return ret

class Calculator():
    def __init__(self):
        self.convs = []
        self.conv1 = Conv2d(3, 32, 8, stride=2)
        self.conv2 = Conv2d(32, 32, 4, stride=2)
        self.conv3 = Conv2d(32, 32, 4, stride=2)
        self.conv4 = Conv2d(32, 32, 4, stride=1)

        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]

    def calculate(self, input):
        for c in self.convs:
            input = c(input)
        return input

input_layer = (3,64,64)
calculator = Calculator()
print(calculator.calculate(input_layer))