from flask import Flask ,render_template , request
import pickle
app = Flask(__name__)

file =open('model.pkl','rb')
clf = pickle.load(file)
file.close()
@app.route('/', methods =["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        ts_val = int(myDict['ts_val'])
        rst_grp = int(myDict['rst_grp'])
        num_deeds = int(myDict['num_deeds'])
        house_income = int(myDict['house_income'])
        time_type = int(myDict['time_type'])
        #Code for inference
        inputFeatures = [ts_val,rst_grp,num_deeds,house_income,time_type]
        timeProb = clf.predict([inputFeatures])[0]
        if timeProb == 0:
            time_int = 'Not Interested!'
        else:
            time_int = 'Intrested!'
        return render_template('show.html', prob = time_int)
    return render_template('index.html')
    #return 'Hello, World!' + str(infProb)

if __name__ == "__main__":
    app.run(debug=True)