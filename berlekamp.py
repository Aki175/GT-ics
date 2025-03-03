# Name: Hakan Bektas

class lfsr:
    def __init__(self, formula):
        self.highest, self.lowest = self.parse(formula)

    def parse(self, formula):
        """
        Parse the formula to get the highest and lowest exponent.
        """
        parts = formula.replace(' ', '').split('+')
        exponents = []
        for part in parts:
            if '^' in part:
                exp = int(part.split('^')[1])
                exponents.append(exp)
            # If there is x which actuallly is x^1, then the lowest value is 1.
            elif 'x' in part:
                exponents.append(1)
        highest = max(exponents)
        lowest = min(exponents)
        print(f"Highest: {highest}, Lowest: {lowest}")
        return highest, lowest

    def initialize(self):
        """
        Initialize the state of the LFSR.
        """
        self.state = [0 for _ in range(self.highest - 1)]
        self.state.append(1)
        print(f"Initial state: {self.state}")

    def generate_output(self):
        """
        Generate the output of the LFSR.
        """
        self.generated_output = []
        for _ in range((2 ** self.highest) - 1):
            output_bit = self.state[-1]
            self.generated_output.append(output_bit)

            val1 = self.state[self.highest - 1]
            val2 = self.state[self.lowest - 1]
            xor = val1 ^ val2

            # Shift 1 to the right, insert xor at the left
            self.state = [xor] + self.state[:-1]

        print(f"LFSR Stream: {self.generated_output}")


def berlekamp_massey(stream):
    """
    Berlekamp-Massey algorithm to find the shortest LFSR (polynomial).
    """
    n = len(stream)
    c = [0] * n
    b = [0] * n
    c[0] = 1
    b[0] = 1
    l = 0
    m = -1

    for i in range(n):
        d = stream[i]
        for j in range(1, l + 1):
            # XOR with previous values based on c
            d ^= c[j] * stream[i - j]
        if d == 1:
            # Keep a copy of c to update b later
            temp = c.copy()
            for j in range(n - i + m):
                c[i - m + j] ^= b[j]
            if 2 * l <= i:
                l = i + 1 - l
                m = i
                # Update b with the previous c
                b = temp

    # Build the polynomial string starting with 1
    poly = "1"
    for i in range(1, l + 1):
        if c[i] == 1:
            poly += f"+x^{i}"
    return poly


if __name__ == '__main__':
    formulas = ['1+x^2+x^3', '1++x^2+x^4' ,'1+x+x^4', '1+x^2+x^5']
    # Create the steams.
    for formula in formulas:
        print(f"\nTesting formula: {formula}")
        l = lfsr(formula)
        l.initialize()
        l.generate_output()
        full_stream = l.generated_output

        # Try to recover the polynomial witht the generated stream.
        polynomial = berlekamp_massey(full_stream)
        print(f"Recovered polynomial: {polynomial}")

        half_stream = full_stream[:len(full_stream) // 2]
        half_polynomial = berlekamp_massey(half_stream)
        print(f"Recovered polynomial (half stream): {half_polynomial}")
