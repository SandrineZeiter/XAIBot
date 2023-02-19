# ----------------------- imports -----------------------
# imports for flask
from PIL.Image import Image
from flask import request, Flask

# imports for ML
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# imports for LIME
from lime.lime_text import LimeTextExplainer

# import matplotlib
import matplotlib

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

# ----------------------- ML-part -----------------------
# import dataset
data = pd.read_csv("/Users/sandrinezeiter/Library/CloudStorage/OneDrive-UniversitédeFribourg/"
                   "Thesis/Twitter_cleaned.csv")

# define the different class names for feelings
class_names = ['afraid', 'alive', 'angry', 'confused', 'depressed', 'good', 'happy',
               'helpless', 'hurt', 'indifferent', 'interested', 'love', 'open', 'positive',
               'sad', 'strong']

# split into training and test set
train, test = train_test_split(data, train_size=0.9)

# tokenize the sentences
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\b[a-zA-Z]{3,}\b', lowercase=False,
                             min_df=5, max_df=0.7, stop_words='english')
vectorizer.fit_transform(train['tweet'])

mnb = MultinomialNB(alpha=0.1)
p1 = make_pipeline(vectorizer, mnb)

alpha_grid = np.logspace(-3, 0, 4)  # Is smoothing parameter for the counts
param_grid = [{'multinomialnb__alpha': alpha_grid}]
gs = GridSearchCV(p1, param_grid=param_grid, cv=5, return_train_score=True)

gs.fit(train.tweet, train.feeling)


# ----------------------- LIME -----------------------

def lime_testing_userinput(userinput):
    # creating explainer
    explainer = LimeTextExplainer(class_names=class_names)
    p1.fit(train.tweet, train.sub_category)
    exp = explainer.explain_instance(userinput, p1.predict_proba, num_features=5, top_labels=len(class_names))
    # print(userinput)
    # get index of the predicted class name
    class_index = exp.available_labels()[0]
    # print("Class index ", class_index)

    prediction = class_names[class_index]
    print("Prediction ", prediction)

    # Get the explanations
    explanations = exp.as_list(label=class_index)
    explanations_as_array = np.array(explanations)
    # print("Explanations: ", explanations)
    print("Explanations as array: ", explanations_as_array)
    # print('Explanation for class %s' % class_names[class_index])
    # print('\n'.join(map(str, exp.as_list(label=class_index))))

    # Create the plot and save it
    exp.show_in_notebook([class_index])
    exp.as_pyplot_figure(label=exp.available_labels()[0])
    figure = plt.savefig("figure.png", bbox_inches="tight")
    # plt.show()

    return prediction, explanations_as_array, figure


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
    fulfillment_image = None
    query_result = req.get('queryResult')
    # print(query_result)

    if query_result.get('action') == 'getname':
        # print("Person, name:", query_result.get('parameters').get('person').get('name'))
        # name = query_result.get('parameters').get('person').get('name')
        if not query_result.get('parameters').get('person').get('name'):
            fulfillment_text = "Hi, nice to meet you. Please, tell me something about your day."

        else:
            name = query_result.get('parameters').get('person').get('name')
            # print(name)
            fulfillment_text = "Hi " + name + ", nice to meet you. Please, tell me something about your day."

    elif query_result.get('action') == 'getinformation':
        userinput = query_result["queryText"]
        text_to_analyze = text_to_analyze + " " + userinput

        # Make sure to get enough data to analyze.
        if len(text_to_analyze) < 200:
            possible_answers = ["Please, tell me more.", "Please, let me know more about that.",
                                "I would like to get to know you better, so, can you tell me more?",
                                "To get to know you better, I still need some information",
                                "Thank you " + name + ". But could you tell me more?"]

            fulfillment_text = random.choice(possible_answers)
            fulfillment_image = None

        else:
            # text_to_analyze = text_to_analyze[:180]
            print(len(text_to_analyze))
            prediction, explanations, figure = lime_testing_userinput(text_to_analyze)
            words = np.array([])
            weights = np.array([])
            for i in range(len(explanations)):
                for k in range(2):
                    # print("Position: ", i, k, explanations[i][k])
                    if k == 0:
                        words = np.append(words, explanations[i][k])
                    else:
                        weights = np.append(weights, explanations[i][k])

            # print("Words:", words)
            # print("Weightings:", weights)

            weights = weights.astype(float)
            negative_idx = np.argwhere(weights < 0)
            positive_idx = np.argwhere(weights > 0)
            negative_words = np.array([])
            positive_words = np.array([])

            if negative_idx.size != 0:
                negative_idx = np.concatenate(negative_idx)
                for i in negative_idx:
                    negative_words = np.append(negative_words, words[i])

                print(negative_words)
                # print(negative_idx)
                for all in negative_idx:
                    print("Negative weights:", weights[all])

            if positive_idx.size != 0:
                positive_idx = np.concatenate(positive_idx)
                for i in positive_idx:
                    positive_words = np.append(positive_words, words[i])

                print(positive_words)
                # print(positive_idx)
                for all in positive_idx:
                    print("Positive weights:", weights[all])

            if negative_idx.size == 0:
                fulfillment_text = "Thank you " + name + " for telling me about your day. " \
                                                         "According to what you said, you feel " + prediction + \
                                   ". The words in favour of my choice for your feeling are " + '"' \
                                   + '", "'.join([str(word) for word in positive_words]) + '"'
            else:
                fulfillment_text = "Thank you " + name + " for telling me about your day. " \
                                                         "According to what you said, you feel " + prediction + \
                                   ". The words in favour of my choice for your feeling are " + '"' \
                                   + '", "'.join([str(word) for word in positive_words]) \
                                   + '". The words that do not support my choice are: ' + '"'\
                                   + '", "'.join([str(word) for word in negative_words]) + '".'
                print(len(fulfillment_text))

            # fulfillment_text = "Thank you " + name + " for telling me about your day. According to what you said, you feel " + prediction + "."
            # fulfillment_image = ("/Users/sandrinezeiter/Library/CloudStorage/OneDrive-UniversitédeFribourg/"
            #       "Thesis/Chatty/XAIBot/figure.png")
            fulfillment_image = "https://upload.wikimedia.org/wikipedia/commons/b/bd/Test.svg"
            text_to_analyze = ''  # So that I don't have to restart the whole script over and over again.
            name = ''

    return {
        "fulfillment_text": fulfillment_text,
        "fulfillment_image": fulfillment_image,
        "source": "webhookdata"
    }


# run the app
if __name__ == '__main__':
    app.run()
