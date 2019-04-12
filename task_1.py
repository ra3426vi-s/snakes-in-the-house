import numpy as np
import matplotlib.pyplot as plt
qpsk_constellation = np.array([complex(1, 1), complex(-1, 1), complex(-1, -1), complex(1, -1)])


def qpsk_modulation(bits):
    """

    :param bits: Integer representation of the bits to be modulated. For BPSK Integer and Binary are the same.
    :type bits: int
    :return: The complex constellation symbol
    :rtype: complex
    """
    try:
        return qpsk_constellation[bits]
    except IndexError:
        raise ValueError('{} is to large for this constellation.'.format(bin(bits)))


def qpsk_demodulation(received_symbol):
    """

    :param received_symbol: The complex symbol to be demodulated.
    :type received_symbol: complex
    :return: The demodulated bits, represented as the corresponding Integer value.
    :rtype: int
    """
    if received_symbol.real < 0 and received_symbol.imag > 0:
        return complex(1, 1)
    elif received_symbol.real < 0 and received_symbol.imag < 0:
        return complex(-1, 1)
    elif received_symbol.real > 0 and received_symbol.imag < 0:
        return complex(-1, -1)
    elif received_symbol.real > 0 and received_symbol.imag > 0:
        return complex(1, -1)


def plot_constellation(constellation, base=4):
    # Extract the x and y values for plotting
    in_phase = [symbol.real for symbol in constellation]
    quadrature = [symbol.imag for symbol in constellation]

    # The code word size, for BPSK: code_word_size k = 1
    # Constellation size: also called alphabet M = 2^k,
    # where k is the number of bits in each symbol (code word size).
    code_word_size = int(np.log2(np.size(constellation)))

    # Plot the constellatoin points
    plt.figure()
    plt.plot(in_phase, quadrature, 'o')
    plt.title("Constellation diagram")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)

    # The following snippet of code just adds strings to the plot
    # with the bits the symbol is representing
    count = 0
    for symbol in constellation:
        if base == 2:
            plt_str = '{0:0{1:d}b}'.format(count, int(code_word_size))
        if base == 10:
            plt_str = '{0:d}'.format(count)
        # x and y cordinates for the string are the real and imaginary parts of the symbol.
        plt.text(symbol.real, symbol.imag, plt_str)
        count += 1

    # Finally show the plot
    plt.show()
