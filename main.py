# ----------------------- imports -----------------------
# imports for flask
import json

from flask import request, Flask, jsonify

# imports for picture
import base64

# imports for pickle to use the train.py
import pickle

# imports for ML
import random


import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np

# imports for LIME
from lime.lime_text import LimeTextExplainer

# import matplotlib
import matplotlib

from PIL import Image
from mimetypes import MimeTypes

mime = MimeTypes()

# imports for daily notification
import requests
import datetime
from datetime import datetime
import time
import schedule

# ----------------------- Definition of the plot -----------------------
# To save the figure
matplotlib.use("Agg")
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('axes', labelsize=16)
matplotlib.rc('figure', titlesize=20)

# ----------------------- Define global variables -----------------------
text_to_analyze = " "
name = " "

# ----------------------- Send notification -----------------------
telegram_token = "5875425343:AAEYkMnnHTmRput4Jmax063dK8bq_GwRw9Q"
telegram_base_url = f"https://api.telegram.org/bot{telegram_token}"


def send_message(chat_id, text):
    url = f"{telegram_base_url}/sendMessage?chat_id={chat_id}&text={text}"
    requests.get(url)
    print(f"Chat ID: {chat_id}")


def daily_message():
    now = datetime.now()
    time_str = now.strftime("%H:%M")
    print(time_str)

    send_time = datetime(now.year, now.month, now.day, 21, 30,0)
    send_time_str = send_time.strftime("%H:%M")
    print(send_time_str)

    if time_str == send_time_str:
        message = "Hi, how are you today?"
        chat_id = '5975577883'  # my chat_id = 5975577883
        send_message(chat_id, message)


#while True:
 #   daily_message()
  #  time.sleep(30)


def make_url(mime_type, bin_data):
    return 'data:' + mime_type + ';base64,' + bin_data


# ----------------------- Setup the trained model -----------------------
# define the different class names for feelings
class_names = ['afraid', 'alive', 'angry', 'confused', 'depressed', 'good', 'happy',
               'helpless', 'hurt', 'indifferent', 'interested', 'love', 'open', 'positive',
               'sad', 'strong']

file = open("Model.train", 'rb')
gs = pickle.load(file)

# ----------------------- LIME -----------------------

# Initialization phase
explainer = LimeTextExplainer(class_names=class_names)


def lime_testing_userinput(userinput):
    # creating explainer
    exp = explainer.explain_instance(userinput,
                                     gs.predict_proba,  # instead of p1.predict_proba
                                     num_samples=5000,
                                     num_features=5,
                                     top_labels=len(class_names))

    # get index of the predicted class name and the prediction
    class_index = exp.available_labels()[0]
    prediction = class_names[class_index]

    # Get the explanations
    explanations = exp.as_list(label=class_index)
    explanations_as_array = np.array(explanations)

    # Create the plot and save it
    exp.show_in_notebook([class_index])
    exp.as_pyplot_figure(label=exp.available_labels()[0])
    figure = plt.savefig("figure.png", bbox_inches="tight")
    # plt.show()
    figure_mimetype = mime.guess_type("figure.png")[0]

    return prediction, explanations_as_array, figure_mimetype
    # , figure


# ----------------------- Flask app -----------------------
# initialize the flask app
app = Flask(__name__)


@app.route('/')
# create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
# define webhook for different actions and results
def webhook():
    global text_to_analyze
    global name

    req = request.get_json(silent=True, force=True)

    fulfillment_text = ''
    # fulfillment_image = None
    query_result = req.get('queryResult')
    # print(query_result)

    action = query_result.get('action')

    output = query_result.get('outputContexts')[0].get('name')
    # print("OutputContext", output)
    name_parts = output.split('/')
    # Session_id should last for 20min
    session_id = '/'.join(name_parts[4:5])
    print("Session ID", session_id)

    app.logger.info(json.dumps(req, indent=4))

    if action == 'getname':

        if not query_result.get('parameters').get('person').get('name'):
            fulfillment_text = "Hi, nice to meet you. Please, tell me something about your day."

        else:
            name = query_result.get('parameters').get('person').get('name')
            name = name.title()
            # print(name)
            fulfillment_text = "Hi " + name + ", nice to meet you. Please, tell me something about your day."
        res = {
            "fulfillmentText": fulfillment_text,
            "source": "webhookdata"
        }
    elif action == 'denial':
        userinput = query_result["queryText"]
        text_to_analyze += " " + userinput
        fulfillment_text = "Too bad, you don't want to tell me anything. Are you really sure?"
        res = {
            "fulfillmentText": fulfillment_text,
            "source": "webhookdata"
        }
    elif action == 'confirmation':
        fulfillment_text = "Thanks anyway for your time, " + name + ". Have a nice day."
        res = {
            "fulfillmentText": fulfillment_text,
            "source": "webhookdata"
        }
    elif action == 'getinformation':
        userinput = query_result["queryText"]
        text_to_analyze += " " + userinput

        # Make sure to get enough data to analyze.
        if len(text_to_analyze) < 256:
            possible_answers = ["Please, tell me more about that.", "Can you provide me more information about that?",
                                "I would like to get to know you better, so, can you tell me more?",
                                "I'm curious to learn more about that",
                                "To get to know you better, I still need some information",
                                "Thanks for sharing, " + name + ". Can you expand on that?"]

            fulfillment_text = random.choice(possible_answers)
            # fulfillment_image = None
            res = {
                "fulfillmentText": fulfillment_text,
                "source": "webhookdata"
            }
        else:
            prediction, explanations, figure_mimetype = lime_testing_userinput(text_to_analyze)

            # with Image.open("figure.png") as img:
            #
            #     max_width = 500
            #     if img.width > max_width:
            #         img.resize((max_width, int(max_width*img.height/img.width)))
            #
            #     img = img.convert("RGB")
            #     img.save("compressed_image.jpg", omptimize=True, quality=85)
            #
            # with open("compressed_image.jpg", 'rb') as f:
            #     image_bytes = f.read()
            #     image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            #
            # print(image_base64[:30])

            with open("figure.png", 'rb') as f:
                data = f.read()
                data64 = base64.b64encode(data).decode('utf-8')
                url = make_url(figure_mimetype, data64)
                print("URL", url)

            # Build the np.arrays for words and weights
            words = explanations[:, 0].flatten()
            weights = explanations[:, 1].flatten()

            # Build np.arrays for positive and negative words
            weights = weights.astype(float)
            negative_idx = np.where(weights < 0)[0]
            positive_idx = np.where(weights > 0)[0]
            negative_words = words[negative_idx]
            positive_words = words[positive_idx]

            # Create fulfillment text for the chatbot
            fulfillment_text = "Thank you {} for telling me about your day. According to what you said, " \
                               "you feel {}. \nThe words in favor of my choice for your feeling are: " \
                               "\"{}\".".format(name, prediction.upper(),
                                                '", "'.join(str(word) for word in positive_words).upper())

            if negative_idx.size > 0:
                if negative_idx.size == 1:
                    fulfillment_text += " \nThe word that would rather indicate a different feeling is: \"{}\".".format(
                        '", "'.join(str(word) for word in negative_words).upper())
                else:
                    fulfillment_text += " \nThe words that would rather indicate a different feeling are: \"{}\".".format(
                        '", "'.join(str(word) for word in negative_words).upper())

            # fulfillment_text = "Thank you " + name + " for telling me about your day. According to what you said, you feel " + prediction + "."
            # fulfillment_image = ("/Users/sandrinezeiter/Library/CloudStorage/OneDrive-UniversitédeFribourg/"
            #       "Thesis/Chatty/XAIBot/figure.png")
            # fulfillment_image = "https://upload.wikimedia.org/wikipedia/commons/b/bd/Test.svg"

            res = {
                #  "fulfillmentText": fulfillment_text,
                "fulfillmentMessages": [
                    {
                        "text": {
                            "text": [
                                fulfillment_text
                            ]
                        },
                        "platform": "TELEGRAM"
                    },
                    {
                        "image": {
                            # "imageUri": url},
                            "imageUri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="},
                        # "imageUri": "https://giphy.com/gifs/love-heart-little-bird-S9oNGC1E42VT2JRysv"},
                        # "imageUri": "figure.png"},
                        # "imageUri":"../imgs/compressed_image.jpg"},#data:image/png;base64," + image_base64
                        "platform": "TELEGRAM"
                    }],
                #     "imageUri":"https://giphy.com/gifs/love-heart-little-bird-S9oNGC1E42VT2JRysv"}     },
                "source": "webhookdata",
            }
            # Set back the global variables

            text_to_analyze = ''
            name = ''

    return jsonify(res)
    # {
    # "fulfillment_text": fulfillment_text,
    # # "fulfillment_image": fulfillment_image,
    # "payload": {
    #     "image": image_base64
    # },
    # res
    # "source": "webhookdata"
    # }


# run the app
if __name__ == '__main__':
    app.run()
