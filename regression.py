import cPickle as pickle
import sys
import json
from collections import defaultdict
import datetime
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
    """ Trains a simple SVR on the input data.
    Uses the hour and the day of week as the regressor variable,
    and the # of logins for that hour as the regressand.
    This reflects the impact on demand in the real world by the hour of day
    and whether it's a weekend or a weekday.
    I could add more dimensions/features to this to further beef it up,
    for example:
        - week number (i.e. 1-52)
        - day of the month (i.e. 1-31)
        - extra external data points like holidays (such as new years eve,
         christmas..), weather, traffic...
    """
    redis = Redis()
    logincount = defaultdict(int)
    for login in logins:
        dt = dateutil.parser.parse(login)
        hour = dt.hour
        day = dt.day
        month = dt.month
        year = dt.year
        logincount[year, month, day, hour] += 1

    # regressor; a tuple of (hour, weekday)
    # the day of the week is an integer, where Monday is 0 and Sunday is 6.
    X = []  # 0,0 1,0, 2,0 ... 20,6 21,6 22,6 23,6

    # regressand; number of logins for that hour of that day of week
    y = [] # 33, 42, 12 ...

    for (year, month, day, hour), numlogins in logincount.iteritems():
        weekday = datetime.datetime(year, month, day, hour).weekday()
        X.append([hour, weekday])
        y.append(numlogins)
    import ipdb; ipdb.set_trace()

    #TODO: Explain C
    svr = svm.SVR(C=15, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                  gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
                  random_state=None, shrinking=True, tol=0.001, verbose=False)
    print 'Fitting...'
    svr.fit(X, y)

    # Save the regressor to redis so we can use it later for predicting.
    redis.set('regressor', pickle.dumps(svr))

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

    mlab.xlabel("day of week")
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