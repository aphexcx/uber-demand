__author__ = 'Afik Cohen'

import sys
import json

try:
    import dateutil.parser
except ImportError:
    print ("You need python-dateutil installed, please run "
           "`pip install python-dateutil`.")
    sys.exit(1)





def train(loginfile):
    with open(loginfile) as f:
        logins = json.load(f)

    for login in logins:
        print login


def main(args):
    train(args[1])

if __name__ == "__main__":
    main(sys.argv)