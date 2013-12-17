__author__ = 'Afik Cohen'

from flask import Flask
from flask import request

from tasks import train_async

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/train', methods=['POST', 'GET'])
def train():
    message = "Error! You need to post json to this endpoint!"
    if request.method == 'POST':
        logins = request.get_json()
        #start training
        result = train_async.delay(logins)
        # import ipdb; ipdb.set_trace()
        message = result.state
    return message


@app.route('/predict')
def predict():
    #get date argument

    #convert to hour, day of month tuple

    #call predict
    return


if __name__ == "__main__":
    app.run()
    from flask import url_for
    print url_for(train)