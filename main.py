__author__ = 'Afik Cohen'

import sys
import json
from collections import defaultdict
from sklearn import svm

try:
    import dateutil.parser
except ImportError:
    print ("You need python-dateutil installed, please run "
           "`pip install python-dateutil`.")
    sys.exit(1)



# >>> from sklearn import svm
# >>> X = [[0, 0], [2, 2]]
# >>> y = [0.5, 2.5]
# >>> clf = svm.SVR()
# >>> clf.fit(X, y)
# SVR(C=1.0, cache_size=200, coef0=0.0, degree=3,
# epsilon=0.1, gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
# random_state=None, shrinking=True, tol=0.001, verbose=False)
# >>> clf.predict([[1, 1]])
# array([ 1.5])

def train(loginfile):
    with open(loginfile) as f:
        logins = json.load(f)

    logincount = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for login in logins:
        dt = dateutil.parser.parse(login)
        hour = dt.hour
        day = dt.day
        month = dt.month
        logincount[month][day][hour] += 1

    # regressor; hour, day of month
    X = [] # 1,1 2,1, 3,1 ... 1,31 2,31 3,31 4,31

    # regressand; number of logins for that hour of that day of month
    y = [] # 33, 42, 12 ...

    for month in [3]:
        #hours_since_start_of_month = 0
        for day in logincount[month]:
            for hour in logincount[month][day]:
                numlogins = logincount[month][day][hour]
                #hours_since_start_of_month += 1
                X.append([hour, day])
                y.append(numlogins)

    #import ipdb; ipdb.set_trace()
    from sklearn import linear_model
    clf = svm.SVR(C=15)
    print 'Fitting...'
    clf.fit(X, y)

    y_clf = clf.predict(X)

    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = Axes3D(fig)

    #x = [6,3,6,9,12,24]
    #y = [3,5,78,12,23,56]
    x = [x[1] for x in X]
    y = y
    z = np.array([x[0] for x in X])
    Xarr, Yarr = np.meshgrid(x,y)

    ax.plot_surface(Xarr, Yarr, z)

    plt.show()

    #import pylab as pl
    #pl.scatter([x[0] for x in X][:], y[:], c='k', label='data')
    #pl.hold('on')
    #predictions = list(clf.predict(X[:]))
    #predictions.insert(0,np.nan)
    #predictions.append(np.nan)
    #pl.plot([np.nan] + [x[0] for x in X][:] + [np.nan], predictions, c='g', label='Model')
    #pl.xlabel('data')
    #pl.ylabel('target')
    #pl.title(str(clf))
    #pl.xlim(0,24)
    #pl.ylim(0,100)
    #pl.legend()
    #pl.show()


    import ipdb; ipdb.set_trace()



def main(args):
    train(args[1])

if __name__ == "__main__":
    main(sys.argv)