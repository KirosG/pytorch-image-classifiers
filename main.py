import argparse


def main():
    pass


if __name__ == "__main__":

    print("Running Model")

    parser = argparse.ArgumentParser(
        description="This is the description..."  # program title
    )

    parser.add_argument(
        'sample',  # argument name
        metavar='n',  #
        type=int,
        help="The number of samples that will be drawn"
    )

    parser.parse_args()

    # parser.print_help()

    main()
