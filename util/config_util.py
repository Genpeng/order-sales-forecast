# _*_ coding: utf-8 _*_

"""
Some utility functions about loading configuration file.

Author: Genpeng Xu
"""

import json
import argparse
from bunch import Bunch


def get_config_from_json(json_file):
    """
    Get the configuration by reading the `json_file` and return a Bunch object,
    which is attribute-style dictionary.

    Argument:
        json_file : str, the path to get configuration file

    Return:
        config : Bunch, an attribute-style dictionary object
        config_dict : dict, a dictionary whose elements represent the names and its corresponding
                      values of configurations
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = Bunch(config_dict)
    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    return config


def get_args():
    parser = argparse.ArgumentParser(description="Get the path of configuration file.")
    parser.add_argument('-c', '--config',
                        dest='config',
                        metavar='config_path',
                        default='None',
                        help="Specify the path to the configuration file.")
    args = parser.parse_args()
    return args, parser


def main():
    print("[INFO] Start parsing configuration file...")

    parser, config = None, None
    try:
        args, parser = get_args()  # get the path of configuration file
        config = process_config(args.config)
    except Exception as e:
        print(e)
        if parser:
            parser.print_help()
        exit(0)

    print(config.learning_rate)
    print(type(config.learning_rate))


if __name__ == '__main__':
    main()
