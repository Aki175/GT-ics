# Andere vak, even snel snel hier gevoegd.

class lfsr:
    def __init__(self, formula):
        self.highest, self.lowest = self.parse(formula)

    def parse(self, formula):
        """
        Parse the formula to get the highest and lowest exponent.
        """
        parts = formula.split('+')
        exponents = []
        for part in parts:
            if '^' in part:
                exp = int(part.split('^')[1])
                exponents.append(exp)
        highest = max(exponents)
        lowest = min(exponents)

        # If there is x which actuallly is x^1, then the lowest value is 1.
        if(highest == lowest):
            lowest = 1
        print(f"Highest: {highest}, Lowest: {lowest}")
        return highest, lowest

    def initialize(self):
        """
        Initialize the state of the LFSR.
        """
        self.state = [0 for i in range(self.highest - 1)]
        self.state.append(1)
        print(self.state)

    def generate_output(self):
        """
        Generate the output of the LFSR.
        """
        # self.generated_output = []
        # for _ in range((2 ** self.highest) - 1):
        #     # Last week indexing
        #     # Get the values from the correct indexes.
        #     val1 = self.state[0]
        #     val2 = self.state[self.highest - self.lowest]
        #     # XOR the two values.
        #     print(val1, val2)
        #     xor = val1 ^ val2
        #     self.generated_output.append(xor)
        #     print(xor)
        #     # Shift everything 1 place to the right and add the xor to the left.
        #     self.state = [xor] + self.state[:-1]
        #     print(self.state)

        """
        Generate the output of the LFSR.
        """
        # self.generated_output = []
        # for _ in range((2 ** self.highest) - 1):
        #     # Get the values from the correct indexes.
        #     val1 = self.state[self.highest - 1]
        #     val2 = self.state[self.lowest - 1]
        #     print(val1, val2)
        #     xor = val1 ^ val2
        #     self.generated_output.append(xor)
        #     print(xor)
        #     # Shift everything 1 place to the right and add the xor to the left.
        #     self.state = [xor] + self.state[:-1]
        #     print(self.state)

        self.generated_output = []
        for _ in range((2 ** self.highest) - 1):
            output_bit = self.state[-1]  # Output is the rightmost bit
            print(f"Output bit: {output_bit}")
            self.generated_output.append(output_bit)

            val1 = self.state[self.highest - 1]
            val2 = self.state[self.lowest - 1]
            xor = val1 ^ val2

            # Shift 1 to the right, insert xor at the left
            self.state = [xor] + self.state[:-1]

        print(f"Generated output: {self.generated_output}")
        print(f"Generated output: {self.generated_output}")


if __name__ == '__main__':
    formula = '1+ x^3+x^5'
    l = lfsr(formula)
    l.initialize()
    l.generate_output()
    print(l.highest, l.lowest)


