import argparse

from array_transfer import check_correctness, time_ak_array_transfer

import arkouda as ak


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure the performance of transferring bigint arrays."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument("-n", "--size", type=int, default=10**6, help="Problem size: length of array")
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=6,
        help="Number of times to run the benchmark",
    )
    parser.add_argument(
        "--max-bits",
        type=int,
        default=-1,
        help="Maximum number of bits, so values > 2**max_bits will wraparound. "
        "-1 is interpreted as no maximum",
    )
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        type=int,
        help="Value to initialize random number generator",
    )
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    ak.verbose = False
    ak.connect(args.hostname, args.port)
    dtype = ak.bigint.name

    if args.correctness_only:
        check_correctness(dtype, args.seed, args.max_bits)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_array_transfer(args.size, args.trials, dtype, args.seed, args.max_bits)
    sys.exit(0)
