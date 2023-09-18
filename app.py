import pyrebase
import mysql.connector

from flask import Flask, render_template, request, redirect, session, send_file
from flask_session import Session

import pickle
import pandas as pd
from sklearn.decomposition import PCA

from sklearn import preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__, template_folder='./templates',
            static_folder='static', static_url_path='/static')

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

firebase = pyrebase.initialize_app({
    "apiKey": "AIzaSyA_S4OHs34DWj3UM7t4eus6QynKBQnwqg0",
    "authDomain": "dogecall-ef1a1.firebaseapp.com",
    "databaseURL": "https://dogecall-ef1a1.firebaseio.com ",
    "storageBucket": "dogecall-ef1a1.appspot.com",
    "serviceAccount": "./json/serviceAccountKey.json"
})

auth = firebase.auth()
storage = firebase.storage()

db = mysql.connector.connect(
    host = "dogecall.cwrriox6nex9.us-east-1.rds.amazonaws.com",
    port = 3306,
    database = "dogecall",
    user = "root",
    password = "rootroot"
)
cursor = db.cursor()

def write_to_mysql(dataframe, connection, table_name):
    # creating column list for insertion
    cols = ",".join([str(i) for i in dataframe.columns.tolist()])

    # Insert DataFrame recrds one by one.
    for i,row in dataframe.iterrows():
        sql = "INSERT INTO " + table_name + "(" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))

        # the connection is not autocommitted by default, so we must commit to save our changes
        connection.commit()

#load all models
pickle_path = ('./models/churnerModel.pkl')
with open(pickle_path, 'rb') as f:
    churners_segm_model = pickle.load(f)

pickle_path = ('./models/nonChurnerModel.pkl')
with open(pickle_path, 'rb') as f:
    non_churners_segm_model = pickle.load(f)

pickle_path = ('./models/predict.pkl')
with open(pickle_path, 'rb') as f:
    pred_model = pickle.load(f)


#flask app routes
@app.route("/")
def main():
    return render_template('index.html')

@app.route("/signOut", methods=['GET', 'POST'])
def signOut():
    return redirect('/')

@app.route("/auth", methods=['GET', 'POST'])
def signIn():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if len(email) < 4 or len(password) < 4:
            return render_template('index.html', error="Email and password must be at least 4 characters.")
        
        try:
            user_auth = auth.sign_in_with_email_and_password(email, password)

            session["email"] = user_auth["email"]
            session["uid"] = user_auth["localId"]
            session["idToken"] = user_auth["idToken"]
            session["displayName"] = user_auth["displayName"]
        
        except:
            return render_template('index.html', error="Invalid password and email combination.")
        
        return redirect('/dashboard')

@app.route("/downloadProfilePic",  methods=['GET'])
def downloadProfilePic():
    userID = session["uid"]

    #download user profile pic
    accessToken = '28095d8a-b3d1-49af-a319-76ed2cc27512'
    return storage.child(userID).get_url(accessToken)

@app.route("/dashboard")
def dashboard():
    img_url = downloadProfilePic()
    print(auth.get_account_info(session["idToken"]))
    return render_template('dashboard.html', img_url=img_url)


@app.route("/userProfile", methods=['GET', 'POST'])
def userProfile():
    img_url = downloadProfilePic()
    #update profile pic
    if request.method == 'POST' and request.form["type"] == 'updateProfilePic':
        img = request.files['profile-pic']
        img_data = img.read()

        #upload picture based on user uid
        storage.child(session["uid"]).put(img_data)
        print('Profile Picture Successfully Uploaded')

    #update name
    elif request.method == 'POST' and request.form["type"] == 'updateName':
        name = request.form["displayName"]
        auth.update_profile(session["idToken"], display_name=name)
        session["displayName"] = name
        print(auth.get_account_info(session["idToken"]))

    return render_template('user-profile.html', img_url=img_url)

#segmentation modelling
@app.route("/segmentation",  methods=['GET', 'POST'])
def segmentation():
    img_url = downloadProfilePic()

    selected_columns = ["Monthly Charges", "Tenure Months"]
    
    #segmentation for churner customers
    if request.method == 'POST' and request.form["type"] == 'segChurnersData':
        file = request.files['file']
        churners_df = pd.read_excel(file)

        #segment customers
        churners_agg_labels = churners_segm_model.fit_predict(churners_df[selected_columns])
        churners_df['Clusters'] = churners_agg_labels

        #renaming the cluster names
        churners_cluster_mapping = {0: 'The Economical Explorer',
                                    1: 'The Loyal High-Value Enthusiasts',
                                    2: 'The Short-Lived Moderate Spenders'}
        
        churners_df['Clusters'] = churners_df['Clusters'].replace(churners_cluster_mapping)

        #fetch all churners data to update the cluster CLTV
        sql_churners_df = pd.read_sql('SELECT * FROM churnersdata', con=db)

        print(sql_churners_df)
        
        churner_cltv_0 = sql_churners_df[sql_churners_df['Clusters'] == 'The Economical Explorer']['ClusterCLTV']
        churner_cltv_1 = sql_churners_df[sql_churners_df['Clusters'] == 'The Loyal High-Value Enthusiasts']['ClusterCLTV']
        churner_cltv_2 = sql_churners_df[sql_churners_df['Clusters'] == 'The Short-Lived Moderate Spenders']['ClusterCLTV']

        churners_cltv =[churner_cltv_0.iloc[0], churner_cltv_1.iloc[0], churner_cltv_2.iloc[0]]
        cluster_names = ['The Economical Explorer', 'The Loyal High-Value Enthusiasts', 'The Short-Lived Moderate Spenders']
        
        # update the "ClusterCLTV" column with the desired value for each clusters
        for name,cltv in zip(cluster_names,churners_cltv):
            churners_df.loc[churners_df["Clusters"] == name, "ClusterCLTV"] = cltv

        churners_df.rename(columns={'Monthly Charges': 'MonthlyCharges',
                            'Tenure Months': 'TenureMonths',
                            }, inplace=True)
        
        #write data to mysql server
        write_to_mysql(churners_df, db, 'churnersdata')
        print("New Customer Data Successfully Uploaded to server and segmented.")

        churners_df.to_excel('./segmentations/segChurners.xlsx', index=False) 
        churners_html = churners_df.to_html(classes='data')

        return render_template('segmentation.html', img_url=img_url,  churnerTable=churners_html)

    #segmentation for non-churner customers
    elif request.method == 'POST' and request.form["type"] == 'segNonChurnersData':
        file = request.files['file']
        non_churners_df = pd.read_excel(file)
        non_churners_df = non_churners_df[selected_columns]

        #segment customers
        pca = PCA(n_components=2)
        pca.fit(non_churners_df)
        non_churners_scores_pca = pca.transform(non_churners_df)
        non_churners_segm_model.fit(non_churners_scores_pca)

        #concat clusters label and components into df
        pca_kmeans_non_churners_df = pd.concat([non_churners_df.reset_index(drop=True), pd.DataFrame(non_churners_scores_pca)], axis=1)
        pca_kmeans_non_churners_df.columns.values[-2:] = ['Component 1', 'Component 2']
        pca_kmeans_non_churners_df['Clusters'] = non_churners_segm_model.labels_

        #renaming the cluster names
        non_churners_cluster_mapping = {0: 'The Premium Lifers',
                                        1: 'The Budget Conscious',
                                        2: 'The Moderate Users',
                                        3: 'The Seasoned Explorers'}
    
        pca_kmeans_non_churners_df['Clusters'] = pca_kmeans_non_churners_df['Clusters'].replace(non_churners_cluster_mapping)

        #fetch all churners data to update the cluster CLTV
        sql_non_churners_df = pd.read_sql('SELECT * FROM nonChurnersdata', con=db)

        print(sql_non_churners_df)
        
        non_churner_cltv_0 = sql_non_churners_df[sql_non_churners_df['Clusters'] == 'The Premium Lifers']['ClusterCLTV']
        non_churner_cltv_1 = sql_non_churners_df[sql_non_churners_df['Clusters'] == 'The Budget Conscious']['ClusterCLTV']
        non_churner_cltv_2 = sql_non_churners_df[sql_non_churners_df['Clusters'] == 'The Moderate Users']['ClusterCLTV']
        non_churner_cltv_3 = sql_non_churners_df[sql_non_churners_df['Clusters'] == 'The Seasoned Explorers']['ClusterCLTV']

        non_churners_cltv =[non_churner_cltv_0.iloc[0], non_churner_cltv_1.iloc[0], non_churner_cltv_2.iloc[0], non_churner_cltv_3.iloc[0]]
        cluster_names = ['The Premium Lifers', 'The Budget Conscious', 'The Moderate Users', 'The Seasoned Explorers']
        
        # update the "ClusterCLTV" column with the desired value for each clusters
        for name,cltv in zip(cluster_names,non_churners_cltv):
            pca_kmeans_non_churners_df.loc[pca_kmeans_non_churners_df["Clusters"] == name, "ClusterCLTV"] = cltv

        pca_kmeans_non_churners_df.rename(columns={'Monthly Charges': 'MonthlyCharges',
                            'Tenure Months': 'TenureMonths',
                            'Component 1': 'Component1',
                            'Component 2': 'Component2',
                            }, inplace=True)
        
        #write data to mysql server
        write_to_mysql(pca_kmeans_non_churners_df, db, 'nonChurnersdata')
        print("New Customer Data Successfully Uploaded to server and segmented.")

        pca_kmeans_non_churners_df.to_excel('./segmentations/nonSegChurners.xlsx', index=False) 
        non_churners_html = pca_kmeans_non_churners_df.to_html(classes='data')

        return render_template('segmentation.html', img_url=img_url,  nonChurnerTable=non_churners_html)

    return render_template('segmentation.html', img_url=img_url)

@app.route("/prediction",  methods=['GET', 'POST'])
def prediction():
    img_url = downloadProfilePic()
    
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_excel(file)

        le = preprocessing.LabelEncoder()

        pred_df = df.drop(['CustomerID','Gender','Phone Service','Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude'],axis = 1)
        columns = ['Senior Citizen','Partner','Dependents', 'Multiple Lines','Internet Service','Online Security','Online Backup',
                'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies','Contract','Paperless Billing','Payment Method' ]

        # Encode Categorical Columns
        pred_df[columns] = pred_df[columns].apply(le.fit_transform)

        predictions = pred_model.predict(pred_df)
        
        predict_result = []
        for i in predictions:
            if i == 1:
                predict_result.append('Yes')
            else:
                predict_result.append('No')

        # Save the DataFrame with predictions to a new Excel file
        df['ChurnLabel'] = predict_result
        df.rename(columns={'Monthly Charges': 'MonthlyCharges',
                            'Tenure Months': 'TenureMonths',
                            'Phone Service': 'PhoneService',
                            'Zip Code': 'ZipCode',
                            'Lat Long': 'LatLong',
                            'Senior Citizen': 'SeniorCitizen',
                            'Multiple Lines': 'MultipleLines',
                            'Senior Citizen': 'SeniorCitizen',
                            'Internet Service': 'InternetService',
                            'Online Security': 'OnlineSecurity',
                            'Online Backup': 'OnlineBackup',
                            'Device Protection': 'DeviceProtection',
                            'Tech Support': 'TechSupport',
                            'Device Protection': 'DeviceProtection',
                            'Streaming TV': 'StreamingTV',
                            'Streaming Movies': 'StreamingMovies',
                            'Paperless Billing': 'PaperlessBilling',
                            'Payment Method': 'PaymentMethod',
                            'Total Charges': 'TotalCharges',
                            }, inplace=True)
        
        df.to_excel('./predictions/churnPrediction.xlsx', index=False) 

        #write data to mysql server
        write_to_mysql(df, db, 'customer')
        print("The customer data successfully uploaded to server.")

        return render_template('prediction.html', img_url=img_url, predictions = predict_result)

    return render_template('prediction.html', img_url=img_url)

@app.route("/sentiment", methods = ['GET', 'POST'])
def sentiment():
    img_url = downloadProfilePic()
    if request.method == 'POST':
        text = request.form['text']
        
        analyzer = SentimentIntensityAnalyzer()

        vader_scores = analyzer.polarity_scores(text)
        vader_polarity = vader_scores['pos'] - vader_scores['neg']
        if vader_polarity > 0:
            vader_sentiment = 'Positive'
        elif vader_polarity < 0:
            vader_sentiment = 'Negative'
        else:
            vader_sentiment = 'Neutral'
        return render_template('sentiment.html',img_url = img_url, sentiment = vader_sentiment)
    
    return render_template('sentiment.html', img_url = img_url)

#app route for excel file downloads
@app.route('/downloadSegChurners')
def download_segChurners():
    return send_file('./segmentations/segChurners.xlsx', as_attachment=True)

@app.route('/downloadNonSegChurners')
def download_nonSegChurners():
    return send_file('./segmentations/nonSegChurners.xlsx', as_attachment=True)

@app.route('/download/predictions')
def download_predictions():
    return send_file('./predictions/churnPrediction.xlsx', as_attachment=True)

#run flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
