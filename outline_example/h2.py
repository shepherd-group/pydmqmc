#!/usr/bin/env python

import pydmqmc as dm


def main() -> None:

    calc = dm.Calculation(
        matrix_file='../systems/EQUILIBRIUM-H4-STO3G.hamil',
        iscomplex=False,
    )

    calc.run()

    return


if __name__ == '__main__':
    main()
