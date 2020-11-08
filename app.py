from flask import Flask,render_template,session,url_for,redirect
import numpy as np 
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
import joblib 


def return_prediction(model,scaler,sample_json):
    
    pre = sample_json['pre']
    glu = sample_json['glu']
    blood = sample_json['blood']
    skin = sample_json['skin']
    insu = sample_json['insu']
    bmi = sample_json['bmi']
    dpf = sample_json['dpf']
    age = sample_json['age']
    
    values = [[pre,glu,blood,skin,insu,bmi,dpf,age]]
    
    values = scaler.transform(values)
    
    classes = np.array(['No Diabetes','Diabetes'])
    
    class_ind = model.predict(values)[0]
    
    return classes[class_ind]



app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'



class diabetes_Form(FlaskForm):

	pre = TextField("Pregnancies")
	glu = TextField("Glucose")
	blood = TextField("BloodPressure")
	skin = TextField("SkinThickness")
	insu = TextField("Insulin")
	bmi = TextField("BMI")
	dpf = TextField("DiabetesPedigreeFunction")
	age = TextField("Age")

	submit = SubmitField("Analyze")





@app.route("/",methods=['GET','POST'])
def index():

	form = diabetes_Form()

	if form.validate_on_submit():

		session['pre'] = form.pre.data
		session['glu'] = form.glu.data
		session['blood'] = form.blood.data
		session['skin'] = form.skin.data 
		session['insu'] = form.insu.data 
		session['bmi'] = form.bmi.data 
		session['dpf'] = form.dpf.data 
		session['age'] = form.age.data 
		
		return redirect(url_for("prediction"))

	return render_template('home.html',form=form)
	


model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('diabetes_scale.pkl')


@app.route('/prediction')
def prediction():
	
	content = {}

	content['pre'] = int(session['pre'])
	content['glu'] = float(session['glu'])
	content['blood'] = float(session['blood'])
	content['skin'] = float(session['skin'])
	content['insu'] = float(session['insu'])
	content['bmi'] = float(session['bmi'])
	content['dpf'] = float(session['dpf'])
	content['age'] = float(session['age'])

	results = return_prediction(model,scaler,content)

	return render_template('prediction.html',results=results)


if __name__ == "__main__":
    app.run(debug=True)