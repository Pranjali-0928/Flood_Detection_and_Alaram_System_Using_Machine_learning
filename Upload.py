from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import numpy as np
from sklearn import  linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC,SVR
import smtplib, ssl


#--------load data set-------------------------
filename = "kerala.csv"
df = pd.read_csv(filename,sep=r'\s*,\s*')
#-------------------show data------------------
# print(df.info())
# print("Annual rainfall",df[' ANNUAL RAINFALL'])
# print("Flood status",df['FLOODS'])
X_train=df['ANNUAL RAINFALL']
X_test=X_train[81:]
X_train=X_train[0:80]
X_train = X_train.values.reshape((-1,1))
X_test = X_test.values.reshape((-1,1))
# print(X_train)
df['FLOODS'].replace({'YES':1,'NO':0},inplace=True)
# print("Flood status",df['FLOODS'])
y_train=df['FLOODS']
y_test=y_train[81:]
y_train=y_train[0:80]
y_train = y_train.values.reshape((-1,1))
y_test = y_test.values.reshape((-1,1))
# print(y_train)
# print(y_test)

app = Flask(__name__)  
@app.route('/Reg')
def Reg():
    ########################### Regression ####################
    y1_train=df['ANNUAL RAINFALL']
    x_train=range(len(y1_train))
    y1_train = y1_train.values.reshape((-1,1))
    x_train = np.reshape(x_train, (-1, 1))
    reg1 = linear_model.LinearRegression()
    reg1.fit(x_train,y1_train)
    x_test=len(y1_train)+1
    x_test = np.reshape(x_test, (-1, 1))
    result= reg1.predict(x_test)
     
######################### classification  ####################

    log_reg = LogisticRegression(random_state=0)
    log_reg.fit(X_train,y_train)
    pred=log_reg.predict(result)
    if(pred):
        pred="Yes" 
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "alertflooddetection@gmail.com"  # Enter your address
        receiver_email = "pranjalipatil9420@gmail.com"  # Enter receiver address
        password = "Flood@123"
        message = """\
        Subject: Flood Alert

        Regression - Yes.
        Gather emergency supplies, including food and water. Store at least 1 gallon of water per day for each person and each pet. Store at least a 3-day supply.
        Listen to your local radio or television station for updates.
        Have immunization records handy (or know the year of your last tetanus shot).
        Store immunization records in a waterproof container.
        Bring in outdoor possessions (lawn furniture, grills, trash cans) or tie them down securely.
        If evacuation appears necessary, turn off all utilities at the main power switch and close the main gas valve.
        Leave areas subject to flooding such as low spots, canyons, washes, etc. (Remember: avoid driving through flooded areas and standing water)"""

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
    else:
        pred="No"    
    return render_template("page3.html",d1='Regression',d2=round(result[0][0]),d3=pred)
@app.route('/Knn')
def Knn():
    ################################# Knn Regression#########################333333
    y1_train=df['ANNUAL RAINFALL']
    x_train=range(len(y1_train))
    y1_train = y1_train.values.reshape((-1,1))
    x_train = np.reshape(x_train, (-1, 1))
    knn1 = KNeighborsRegressor()
    knn1.fit(x_train,y1_train)
    x_test=len(y1_train)+1
    x_test = np.reshape(x_test, (-1, 1))
    result= knn1.predict(x_test)
    print(result)
    ################################### Knn classification########################33
    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)
    pred=knn.predict(result)
    if(pred):
        pred="Yes"
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "alertflooddetection@gmail.com"  # Enter your address
        receiver_email = "pranjalipatil9420@gmail.com"  # Enter receiver address
        password = "Flood@123"
        message = """\
        Subject: Flood Alert

        Knn - Yes.
        Gather emergency supplies, including food and water. Store at least 1 gallon of water per day for each person and each pet. Store at least a 3-day supply.
        Listen to your local radio or television station for updates.
        Have immunization records handy (or know the year of your last tetanus shot).
        Store immunization records in a waterproof container.
        Bring in outdoor possessions (lawn furniture, grills, trash cans) or tie them down securely.
        If evacuation appears necessary, turn off all utilities at the main power switch and close the main gas valve.
        Leave areas subject to flooding such as low spots, canyons, washes, etc. (Remember: avoid driving through flooded areas and standing water)"""

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
    else:
        pred="No" 
    return render_template("page3.html",d1='Knn',d2=round(result[0][0]),d3=pred)
@app.route('/Random')
def Random():
    ############################### RandomForest Regression############################
    y1_train=df['ANNUAL RAINFALL']
    x_train=range(len(y1_train))
    y1_train = y1_train.values.reshape((-1,1))
    x_train = np.reshape(x_train, (-1, 1))
    clf1= RandomForestRegressor()
    clf1.fit(x_train,y1_train)
    x_test=len(y1_train)+1
    x_test = np.reshape(x_test, (-1, 1))
    result= clf1.predict(x_test)
    print(result)
    ############################### RandomForest classification ############################
    clf= RandomForestClassifier()
    clf.fit(X_train,y_train)
    pred=clf.predict([result])
    if(pred):
        pred="Yes"
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "alertflooddetection@gmail.com"  # Enter your address
        receiver_email = "pranjalipatil9420@gmail.com"  # Enter receiver address
        password = "Flood@123"
        message = """\
        Subject: Flood Alert

        RandomForest - Yes.
        Gather emergency supplies, including food and water. Store at least 1 gallon of water per day for each person and each pet. Store at least a 3-day supply.
        Listen to your local radio or television station for updates.
        Have immunization records handy (or know the year of your last tetanus shot).
        Store immunization records in a waterproof container.
        Bring in outdoor possessions (lawn furniture, grills, trash cans) or tie them down securely.
        If evacuation appears necessary, turn off all utilities at the main power switch and close the main gas valve.
        Leave areas subject to flooding such as low spots, canyons, washes, etc. (Remember: avoid driving through flooded areas and standing water)"""

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
    else:
        pred="No" 
    return render_template("page3.html",d1='RandomForest',d2=round(result[0]),d3=pred)
@app.route('/svm')
def svm():
    ############################### RandomForest Regression############################
    y1_train=df['ANNUAL RAINFALL']
    x_train=range(len(y1_train))
    y1_train = y1_train.values.reshape((-1,1))
    x_train = np.reshape(x_train, (-1, 1))
    svr= SVR()
    svr.fit(x_train,y1_train)
    x_test=len(y1_train)+1
    x_test = np.reshape(x_test, (-1, 1))
    result= svr.predict(x_test)
    print(result)
    ############################### RandomForest classification ############################
    svc=SVC()
    svc.fit(X_train,y_train)
    pred=svc.predict([result])
    if(pred):
        pred="Yes"
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "alertflooddetection@gmail.com"  # Enter your address
        receiver_email = "pranjalipatil9420@gmail.com"  # Enter receiver address
        password = "Flood@123"
        message = """\
        Subject: Flood Alert

        SVM - Yes.
        Gather emergency supplies, including food and water. Store at least 1 gallon of water per day for each person and each pet. Store at least a 3-day supply.
        Listen to your local radio or television station for updates.
        Have immunization records handy (or know the year of your last tetanus shot).
        Store immunization records in a waterproof container.
        Bring in outdoor possessions (lawn furniture, grills, trash cans) or tie them down securely.
        If evacuation appears necessary, turn off all utilities at the main power switch and close the main gas valve.
        Leave areas subject to flooding such as low spots, canyons, washes, etc. (Remember: avoid driving through flooded areas and standing water)"""

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
    else:
        pred="No" 
    
    return render_template("page3.html",d1='SVM',d2=round(result[0]),d3=pred)           
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        print(request.form['username'])
        print(request.form['password'])
        if request.form['username'] =='admin' and  request.form['password'] =='admin':
            log_reg = LogisticRegression(random_state=0)
            log_reg.fit(X_train,y_train)
            #  ## Evaluating the model
            log_reg = round(log_reg.score(X_test,y_test)*100)
            # print("Log_reg_score",log_reg)
            ## Build an model (KNN)
            knn = KNeighborsClassifier()
            knn.fit(X_train,y_train)
            ## Evaluating the model
            knn = round(knn.score(X_test,y_test)*100)
            # print("knn_score",knn)
            ## Build an model (Random forest classifier)
            clf= RandomForestClassifier()
            clf.fit(X_train,y_train)
            ## Evaluating the model
            clf = round(clf.score(X_test,y_test)*100)
            # print("clf_score",clf)
            ## Build an model (Support Vector Machine)
            svm = SVC()
            svm.fit(X_train,y_train)
            ## Evaluating the model
            svm = round(svm.score(X_test,y_test)*100)
            # print("SVM_score",svm)
            return render_template("page2.html",d1=log_reg,d2=knn,d3=clf,d4=svm)
            
        else:
            error = 'Invalid Credentials. Please try again.'
            
    return render_template('login.html', error=error)
# @app.route('/upload')  
# def upload():  
#     return render_template("file_upload_form.html")  
 

if __name__ == '__main__':
    # app.run(host='192.168.43.84',port=5000)
    app.run(debug = True)
