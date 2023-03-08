from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms.validators import DataRequired, ValidationError
from wtforms.fields import StringField, SubmitField, FileField
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
app = Flask(__name__)
#parameters = []
parameters = pd.DataFrame




class HppInputForm(FlaskForm):

    my_file = FileField('csv_file', validators=[FileRequired(), FileAllowed(['csv','json'])])



    predict = SubmitField('Submit Form')


@app.route("/")
def hello():
    logger.debug("Home page")
    return render_template("index.html")


# HPP Page
@app.route('/hpp_input', methods=['GET', 'POST'])
def hpp_input():
    form = HppInputForm()

    if form.predict.data and form.validate_on_submit():
        logger.debug("Processing user inputs")
        global parameters
        if ".csv" in form.my_file.data.filename:
        
            parameters = pd.read_csv(form.my_file.data)
        else:
            
            parameters = pd.read_json(form.my_file.data)
            parameters = parameters.astype({"Date": str })
        logger.debug("User inputs taken")
        return render_template("display_prediction.html")
    return render_template("hpp_databroker.html", example_form=form)


def get_parameters():
    logger.debug("Return databroker parameters")
    return parameters


def app_run():
    app.secret_key = "hpp"
    bootstrap = Bootstrap(app)
    app.run(host="0.0.0.0", port=8062)
    # app.run()
