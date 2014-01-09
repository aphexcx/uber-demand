import cPickle as pickle
import sys
import json
from collections import defaultdict
from redis import Redis
from sklearn import svm

try:
    import dateutil.parser
except ImportError:
    print ("You need python-dateutil installed, please run "
           "`pip install python-dateutil`.")
    sys.exit(1)

import numpy as np
from mayavi import mlab


class UntrainedException(Exception):
    pass


class PlotException(Exception):
    pass


#TODO: Day of week
def train(logins):
    redis = Redis()
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

    #TODO: why is this 3?
    for month in [3]:
        #hours_since_start_of_month = 0
        for day in logincount[month]:
            for hour in logincount[month][day]:
                numlogins = logincount[month][day][hour]
                #hours_since_start_of_month += 1
                X.append([hour, day])
                y.append(numlogins)

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
    svr = svm.SVR(C=15)
    print 'Fitting...'
    svr.fit(X, y)

    redis.set('regressor', pickle.dumps(svr))
    #
    #y_clf = clf.predict(X)

    #from mpl_toolkits.mplot3d import Axes3D
    #import numpy as np
    #import matplotlib
    #import matplotlib.pyplot as plt

    #fig = plt.figure()
    #ax = Axes3D(fig)

    #x = [6,3,6,9,12,24]
    #y = [3,5,78,12,23,56]

    ##
    #x = [x[1] for x in X]
    #y = y
    #z = [x[0] for x in X]
    #csv = []
    #for i, x in enumerate(X):
    #    csv.append("%s,%s,%s\n" % (x[0], x[1], y[i]))
    #
    #with open("axes.csv", "w") as f:
    #    f.writelines(csv)

    x = np.array([tup[1] for tup in X])
    y = np.array(y)
    z = np.array([tup[0] for tup in X])

    redis.set('x', pickle.dumps(x))
    redis.set('y', pickle.dumps(y))
    redis.set('z', pickle.dumps(z))


def plot(view="iso"):
    redis = Redis()
    x, y, z = (pickle.loads(redis.get('x')), pickle.loads(redis.get('y')),
               pickle.loads(redis.get('z')))
    if None in (x, y, z):
        raise UntrainedException("You must train first!")

    # mlab.options.offscreen = True

    # Simple plot.
    fig = mlab.figure(size=(800, 600))
    # fig.scene.off_screen_rendering = True
    # Define the points in 3D space
    # including color code based on Z coordinate.
    mlab.points3d(x, y, z, y)

    mlab.xlabel("day of month")
    mlab.ylabel("# logins")
    mlab.zlabel("hour")

    views = {"xp": fig.scene.x_plus_view,
             "xm": fig.scene.x_minus_view,
             "yp": fig.scene.y_plus_view,
             "ym": fig.scene.y_minus_view,
             "zp": fig.scene.z_plus_view,
             "zm": fig.scene.z_minus_view,
             "iso": fig.scene.isometric_view
    }

    try:
        views[view]()
    except KeyError as e:
        raise PlotException("Invalid viewwwww option: %s" % view)

    # can't save directly to stringIO, so have to go through a file
    fig.scene.save_png('fig.png')
    fig.parent.close_scene(fig)
    mlab.show()  # cycle mlab's event loop to close it down
    with open('fig.png', 'rb') as f:
        buf = f.read()

    return buf


def predict(tuple_list):
    redis = Redis()
    if not redis.exists('regressor'):
        raise UntrainedException("You must train first!")
    reg = pickle.loads(redis.get('regressor'))
    return reg.predict(tuple_list)


def main(args):
    with open(args[1]) as f:
        logins = json.load(f)
    train(logins)


if __name__ == "__main__":
    main(sys.argv)