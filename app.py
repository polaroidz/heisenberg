import json
from flask import Flask
from flask import request
import heisenberg
import numpy as np

app = Flask(__name__)

sent_size = 30

@app.route("/", methods=['GET'])
def hello():
    return "Jesse!"

@app.route("/", methods=['POST'])
def webhook():
    model = heisenberg.get_model(pretrained=True)

    event = request.data
    event = json.loads(event)

    query = event['result']['resolvedQuery']
    query = query.lower()
    query = heisenberg.clear_string(query)
    
    pred_sent = query + " "

    print(pred_sent)

    for i in range(200):
        X = map(heisenberg.v2k.get, pred_sent[-sent_size:])
        X = np.array(list(X)).reshape(1, -1)

        y = model.predict(X)
        c = heisenberg.vec2str(y)

        pred_sent += heisenberg.vec2str(y)

        if c == '.':
            break

    res = {
        'source': 'Heisenberg',
        'speech': pred_sent,
        'displayText': pred_sent
    }

    return json.dumps(res)

app.run(debug=True)