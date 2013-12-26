import sys

try:
    import dateutil.parser
except ImportError:
    print ("You need python-dateutil installed, please run "
           "`pip install python-dateutil`.")
    sys.exit(1)
from dateutil import rrule
from uuid import uuid4
from redis import Redis

# posting:
# "C:\Program Files\cURL\bin\curl.exe" http://127.0.0.1:5000/train -H "Content-Type: application/json" -X POST --data @uber_demand_prediction_challenge.json
# run celery worker first:
# celery worker --app=tasks --loglevel=info

from flask import Flask
from flask import request
from flask import session
from flask import make_response

from redis_session import RedisSessionInterface
from tasks import train_async
import regression

FLASK_SECRET_KEY = '88zcH3lRhoMVjdoxaK3o'

app = Flask(__name__)
# app.session_interface = RedisSessionInterface()


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/train', methods=['POST', 'GET'])
def train():
    message = "Error! You need to post json to this endpoint!"
    if request.method == 'POST':
        redis = Redis()
        # if redis.exists('regressor'):
        #     message = "Re-training!"
        # else:
        message = "Training new regressor!"
        logins = request.get_json()
        #start training
        result = train_async.delay(logins)
        # import ipdb; ipdb.set_trace()

    return message


@app.route('/predict', methods=['GET'])
def predict():
    #get date argument
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    #convert to hour, day of month tuples
    start_dt = dateutil.parser.parse(start_date)
    end_dt = dateutil.parser.parse(end_date)

    input_range = []
    for dt in rrule.rrule(rrule.HOURLY, dtstart=start_dt, until=end_dt):
        input_range.append([dt.hour, dt.day])

    #call predict
    try:
        prediction_array = regression.predict(input_range)
    except regression.PredictionException:
        return "Error: please train the regressor first! Call the /train endpoint."

    csv = []
    for i, dt in enumerate(rrule.rrule(rrule.HOURLY, dtstart=start_dt, until=end_dt)):
        csv.append(",".join([str(dt), str(prediction_array[i])]))

    csv_response = make_response("\n".join(csv))
    csv_response.headers["content-type"] = "text/csv"
    return csv_response


@app.route('/plot')
def plot():
    # create plot pngs in different orientations
    # show them in a page?
    return

if __name__ == "__main__":
    if sys.argv[-1] == '-d':
        app.debug = True
    app.secret_key = FLASK_SECRET_KEY
    app.run()