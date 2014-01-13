"""Uber Demand Prediction
https://github.com/aphexcx
aphex@aphex.cx

One of the most important signals in Uber's system is demand. It's used in
a variety of capacities, from calculating dynamic (surge) pricing to making
decisions about the number of drivers needed to keep the system running
smoothly. One of the ways Uber quantifies demand is by tracking when users
open the Uber app on their phones.

This service predicts future demand based on historical data.

Setup:
1) Install MayaVi:
    a) Grab VTK and wxPython:
        apt-get python-vtk python-wxgtk2.8 (debian/ubuntu)
        yum install vtk-python wxPython-devel (centos/rhel)
    b) On Centos, mayavi works better if you install it from yum:
        yum install Mayavi
       Otherwise, just use pip:
        pip install wxpython mayavi

2) Install requirements.txt
        pip install -r requirements.txt

3) Start redis:
        redis-server
    and run the celery worker with:
        celery worker --app=tasks --loglevel=info

You're ready to run!
        python main.py

"""
import logging
import pydoc
import sys

import dateutil.parser
from dateutil import rrule

from flask import Flask
from flask import request
from flask import make_response

from tasks import train_async, plot_async
import regression

_FLASK_SECRET_KEY = '88zcH3lRhoMVjdoxaK3o'

app = Flask(__name__)


@app.route("/")
def index():
    """/
    Main index page. Welcome!"""
    html_doc = pydoc.HTMLDoc()
    html = html_doc.docmodule(sys.modules[__name__])
    return html


@app.route('/train', methods=['POST', 'GET'])
def train():
    """/train
    Accepts training data. Example usage with curl:
    curl http://127.0.0.1:5000/train -H "Content-Type: application/json" -X POST --data @uber_demand_prediction_challenge.json

    Data should be a JSON encoded object in the form of a list of UTC timestamps.
    Each timestamp indicates a login.

    Example:
    ["2012-03-01T00:05:55+00:00", "2012-03-01T00:06:23+00:00", "2012-03-01T00:06:52+00:00", ... , "2012-04-30T23:59:29+00:00"]
    """
    message = "Error! You need to post JSON to this endpoint!"
    if request.method == 'POST':
        logins = request.get_json()
        if type(logins) is not list:
            return "Error, valid JSON contains a list of timestamps!"
        #start training
        train_async.delay(logins)
        message = "Training new regressor!"

    return message


@app.route('/predict', methods=['GET'])
def predict():
    """/predict
    Returns predictions, in CSV format.

    The first column is an ISO-formatted UTC timestamp of the start of the
    prediction period and the second column is the predicted # of logins.
    For example, a few rows of predictions might look like:
        2012-05-01 00:00:00,19.1158177963
        2012-05-01 01:00:00,22.0997300016
        2012-05-01 02:00:00,26.1003343227

    Example usage:
    curl -X GET "http://127.0.0.1:5000/predict?start_date=2012-05-01&end_date=2012-05-15"
    """
    #get date argument
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    #convert to hour, weekday tuples
    start_dt = dateutil.parser.parse(start_date)
    end_dt = dateutil.parser.parse(end_date)

    input_range = []
    for dt in rrule.rrule(rrule.HOURLY, dtstart=start_dt, until=end_dt):
        input_range.append([dt.hour, dt.weekday()])

    #call predict
    try:
        prediction_array = regression.predict(input_range)
    except regression.UntrainedException:
        return "Error: please train the regressor first! Call the /train endpoint."

    csv = []
    for i, dt in enumerate(rrule.rrule(rrule.HOURLY, dtstart=start_dt, until=end_dt)):
        csv.append(",".join([str(dt), str(prediction_array[i])]))

    csv_response = make_response("\n".join(csv))
    csv_response.headers["content-type"] = "text/csv"
    return csv_response


@app.route('/plot')
def plot():
    """/plot
    Returns 3D plots of the training data, in PNG format.

    Example usage:
    curl -X GET "http://127.0.0.1:5000/plot?view=iso"

    Examples of possible valid view orientations:
        "http://127.0.0.1:5000/plot?view=xp": x axis, plus
        "http://127.0.0.1:5000/plot?view=xm": x axis, minus
        "http://127.0.0.1:5000/plot?view=yp": y axis, plus
        "http://127.0.0.1:5000/plot?view=ym": y axis, minus
        "http://127.0.0.1:5000/plot?view=zp": z axis, plus
        "http://127.0.0.1:5000/plot?view=zm": z axis, minus
        "http://127.0.0.1:5000/plot?view=iso": isometric
    """
    view = request.args.get("view", "iso")
    # delegating to celery because mayavi's plotting can't be done
    # on a flask thread.
    # fig = regression.plot(view)
    result = plot_async.delay(view)
    try:
        fig = result.get()
    except regression.UntrainedException:
        return "Error: please train the regressor first! Call the /train endpoint."
    png_response = make_response(fig)
    png_response.mimetype = 'image/png'
    return png_response


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    app.secret_key = _FLASK_SECRET_KEY

    if '-d' in sys.argv:
        app.debug = True
    if '-e' in sys.argv:
        # externally visible server
        logger.info('Service is externally available.')
        app.run(host='0.0.0.0')
    else:
        # only available from 127.0.0.1
        logger.info('Service only locally available.')
        app.run()
