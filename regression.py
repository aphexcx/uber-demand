# -*- coding: utf-8 -*-

import cPickle as pickle
import sys
import json
from collections import defaultdict
import datetime
import logging

import dateutil.parser
from redis import Redis
from sklearn import svm
import numpy as np
from mayavi import mlab


class UntrainedException(Exception):
    pass


class PlotException(Exception):
    pass


def train(logins):
    """ Trains sklearn's epsilon-Support Vector Regression model on input data.
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
    logger = logging.getLogger(__name__)
    redis = Redis()
    logincount = defaultdict(int)
    for login in logins:
        dt = dateutil.parser.parse(login)
        hour = dt.hour
        day = dt.day
        month = dt.month
        year = dt.year
        logincount[year, month, day, hour] += 1

    # regressor; tuples of (hour, weekday)
    # the day of the week is an integer, where Monday is 0 and Sunday is 6.
    X = []  # (0,0), (1,0), (2,0), ... (21,6), (22,6), (23,6)

    # regressand; number of logins for that hour of that day of week
    y = []  # 33, 42, 12, ...

    for (year, month, day, hour), numlogins in logincount.iteritems():
        weekday = datetime.datetime(year, month, day, hour).weekday()
        X.append([hour, weekday])
        y.append(numlogins)

    # import ipdb; ipdb.set_trace()

    # Generating a good value for C:
    # from https://icme.hpc.msstate.edu/mediawiki/images/5/55/SVR.pdf
    # C is also referred to as the regression parameter or penalty parameter.
    # Cherkassky and Ma propose C be chosen as
    # C = max(|µy + 3σy)|, |µy - 3σy)|)
    # where µy and σy are the mean and standard deviation of the training point responses.
    C = max(abs(np.mean(y) + 3*np.std(y)), abs(np.mean(y) - 3 * np.std(y)))

    svr = svm.SVR(C=C, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                  gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
                  random_state=None, shrinking=True, tol=0.001, verbose=False)

    # Fit our SVR to the training data.
    logger.info('Fitting...')
    svr.fit(X, y)

    # Save the regressor to redis so we can use it later for predicting.
    redis.set('regressor', pickle.dumps(svr))

    # Generate the x y z coords to be used for plotting.
    x = np.array([tup[1] for tup in X])
    y = np.array(y)
    z = np.array([tup[0] for tup in X])

    redis.set('x', pickle.dumps(x))
    redis.set('y', pickle.dumps(y))
    redis.set('z', pickle.dumps(z))


def plot(view="iso"):
    redis = Redis()
    px, py, pz = (redis.get('x'), redis.get('y'), redis.get('z'))
    if None in (px, py, pz):

        raise UntrainedException("You must train first!")

    x, y, z = (pickle.loads(px), pickle.loads(py), pickle.loads(pz))

    fig = mlab.figure(size=(800, 600))

    # these options could help on certain platforms
    # mlab.options.offscreen = True
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
        raise PlotException("Invalid view option: %s" % view)
    # import ipdb; ipdb.set_trace()
    # can't save directly to stringIO, so have to go through a file
    fig.scene.save_png('fig.png')
    # mayavi doesn't seem to play well with celery on some platforms and
    # doesn't shut down properly - probably because it's in a background thread
    # celery just throws a WorkerLostError on centos.
    # fig.remove()

    # fig.parent.close_scene(fig)
    # this doesn't work on centos:
    # mlab.close()

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
