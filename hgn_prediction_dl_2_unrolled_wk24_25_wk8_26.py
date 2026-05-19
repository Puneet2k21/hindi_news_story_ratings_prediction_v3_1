import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import yaml
import streamlit_authenticator as stauth
import datetime
import gspread
from google.oauth2.service_account import Credentials
import pytz
import logging

# -------------------------------
# 1. Load Authentication Config
# -------------------------------
#This loads all authorized users from a YAML configuration file.
#Opens the YAML file containing login credentials.
#Reads YAML contents and converts them into a Python dictionary

with open("allowed_users.yaml") as file:
    config = yaml.safe_load(file)

# -------------------------------
# 2. Setup Google Sheet for Login Logs
# -------------------------------
#Caches the Google Sheets connection
#Without caching, Google authentication runs repeatedly,slower app, repeated OAuth calls
#With caching, authentication happens once, reused across app sessions


@st.cache_resource

#Creates reusable Google Sheets connection.
def init_google_sheet():

#Defines Google API permissions.
#Allows app to, read Google Sheets, write login records, access Drive-based spreadsheet

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

#Loads secure Google credentials from Streamlit secrets.
#Contains, private key, client email,project ID
#Used for server-to-server authentication

    service_account_info = st.secrets["service_account"]

#Creates authenticated Google credential object.

    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=scope
    )

#Creates authenticated Google Sheets client.
    client = gspread.authorize(creds)

#Opens spreadsheet and worksheet
#Returns worksheet connection for later use

    sheet = client.open("Streamlit_login_track").worksheet("hindi_news_app")

    return sheet

#Function to store login activity.
#Prevents app crashes if Google logging fails
#Gets cached Google Sheet connection.
#Ensures timestamps are recorded in IST
#Creates row to append into Google Sheet
#Adds login record to Google Sheet
#Indicates logging succeeded.


def log_user_login(username):

    try:
        sheet = init_google_sheet()

        ist = pytz.timezone('Asia/Kolkata')

        login_time = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        new_row = [username, login_time]

        sheet.append_row(new_row)

        return True

#Catches all Google/API/network errors.
#Stores error in logs.
#Does NOT crash app
#Signals login tracking failed.

    except Exception as e:

        logging.error(f"Google login logging failed: {e}")

        return False

# -------------------------------
# 3. Login Authentication
# -------------------------------
#Creates Streamlit authentication object.
#Uses users loaded from YAML.
#Session cookie identified, Used to keep user logged in.
#Encrypts login session cookie
#Keeps user logged in for 7 days.

authenticator = stauth.Authenticate(
    config['credentials'],
    'news_app_cookie_test',
    'abc123',
    cookie_expiry_days=7
)

#Displays username and password box
#Validates credentials.

login_result = authenticator.login()

# -------------------------------
# 4. After Successful Login
# -------------------------------
#Runs only if login succeeds.

#Attempts Google login logging.
#If Google logging fails, app continues, prediction engine still works
#Excellent production-safe design 
#Displays logout button.

if st.session_state['authentication_status']:

    # Safe login logging
    login_logged = log_user_login(st.session_state["username"])

    if not login_logged:
        print("Login tracking unavailable")

        # Optional:
        # st.warning("Login tracking temporarily unavailable")

    authenticator.logout()

#Shows logged-in user name.

    st.write(f'Welcome *{st.session_state["name"]}*')

    # App Title
    st.title("Hindi News Story Rating Prediction based on AI model")

    #define dropdown choices visible in UI.
    # ----------------------------------------
    # 5. Define Dropdown Input Options
    # ----------------------------------------
    genre_options = ["ASTROLOGY", "BUSINESS/FINANCE", "CAREER/EDUCATION", "CRIME/LAW & ORDER", "ELECTIONS", "ENTERTAINMENT", 
                 "EVENT/CELEBRATION/AWARDS", "HEALTH", "INDIA-PAK", "INTERNATIONAL", "MISHAPS/FAILURE OF MACHINERY", "MODI VISIT",
                 "NATIONAL THREAT/DEFENCE NEWS", "POLITICAL","RELIGIOUS/FAITH","SCIENCE/SPACE", "SPORTS", "TECHNOLOGY/GADGET",
                 "VIRAL VIDEOS", "WAR", "WEATHER/ENVIRONMENT", "WILD LIFE"]

    geography_options = ["NATIONAL","INTERNATIONAL", "ANDHRA PRADESH/TELANGANA", "BIHAR/JHARKHAND", "DELHI", "GUJARAT", "HIMACHAL PRADESH",  
                     "JAMMU AND KASHMIR/LADAKH", "KARNATAKA", "KERALA", "MADHYA PRADESH/CHHATTISGARH", "MAHARASHTRA/GOA", 
                     "NORTH-EAST", "ODISHA", "PUNJAB/HARYANA/CHANDIGARH","RAJASTHAN", "TAMIL NADU","UTTAR PRADESH/UTTARAKHAND", "WEST BENGAL"]

    popularity_options = ["H", "M", "L"]

    personality_genre_options = ["AAP", "AIMIM", "ASP","ASTROLOGER", "BJP", "BSP", "BUSINESS", "ENTERTAINER", "INC", "INTERNATIONAL", "JDU", "JMM",
                             "LJP", "NC", "NCP", "PDP", "RELIGIOUS", "RJD", "RLD", "RSS-VHP", "SBSP", "SP",
                             "SPORTS", "SS", "TDP", "TMC" ,"OTHERS"]

    logistics_options = ["ON LOCATION", "IN STUDIO", "BOTH"]
    story_format_options = ["DEBATE OR DISCUSSION", "INTERVIEW", "REPORT"]

    
    # ----------------------------------------
    # 6. Input Fields
    #Creates dropdown selection in UI.
    # ----------------------------------------
    genre = st.selectbox("Genre", genre_options)
    geography = st.selectbox("Geography", geography_options)
    personality_popularity = st.selectbox("Personality Popularity (H/M/L)", popularity_options)
    personality_genre = st.selectbox("Personality Genre", personality_genre_options)
    logistics = st.selectbox("Logistics", logistics_options)
    story_format = st.selectbox("Story Format", story_format_options)

    #Everything below runs ONLY after user clicks button

    if st.button("Predict Tier"):

        # ----------------------------------------
        # 7. Prepare DataFrame from Input
        #Converts user selections into structured dataframe.
        #Exactly same format as training data.
        # ----------------------------------------

        new_data = pd.DataFrame({
            'Genre': [genre],
            'Geography': [geography],
            'Personality Popularity': [personality_popularity],
            'Personality-Genre': [personality_genre],
            'Logistics': [logistics],
            'Story_Format': [story_format]
        })

        new_data['Personality Popularity Ord'] = new_data['Personality Popularity'].map({'H': 2, 'M': 1, 'L': 0})

        categorical_columns = ['Genre', 'Geography', 'Personality Popularity', 'Personality-Genre',
                               'Logistics', 'Story_Format']

        def df_to_input_dict(df, columns):
            return {f"{col}_input": df[col].values for col in columns}

        # ----------------------------------------
        # 8. Load Label Encoders
#Loads previously saved encoders from training phase.
#Ensures, same encoding during inference
#prevents category mismatch
#Very important production ML principle.

        # ----------------------------------------
        with open("label_encoders_model3.pkl", "rb") as f:
            label_encoders_model3 = pickle.load(f)
        with open("label_encoders_model4.pkl", "rb") as f:
            label_encoders_model4 = pickle.load(f)

        encoded_data_model3 = new_data.copy()
        encoded_data_model4 = new_data.copy()

#Converts categories into numeric IDs.
#These IDs feed embedding layers.

        for col in categorical_columns:
            encoded_data_model3[col] = label_encoders_model3[col].transform(encoded_data_model3[col])
            encoded_data_model4[col] = label_encoders_model4[col].transform(encoded_data_model4[col])

#Converts dataframe into TensorFlow input format
#Required because model has multiple input layers.

        input_dict_model3 = df_to_input_dict(encoded_data_model3, categorical_columns)
        input_dict_model4 = df_to_input_dict(encoded_data_model4, categorical_columns)

        # ----------------------------------------
        # 9. Load Models

#Loads: 5 DNN models, 5 TabTransformer models

        # ----------------------------------------
        model3_paths = [f"model3_fold{i}_best.keras" for i in range(1, 6)]
        model4_paths = [f"model4_fold{i}_best.keras" for i in range(1, 6)]
        models = [load_model(path) for path in model3_paths + model4_paths]

        # ----------------------------------------
        # 10. Weighted Soft Voting
#This is the core inference engine.
#Each model gets importance based on F1-score performance
#Better models influence prediction more

        # ----------------------------------------
        weights = [0.0994, 0.0947, 0.1052, 0.0993, 0.1023, 0.0991, 0.0960, 0.1035, 0.0989, 0.1015]  # From your F1 scores

#Each model predicts probability distribution.
#Combines predictions from all 10 models
#This is, weighted soft voting ensemble.

        soft_preds = None
        for model, weight, input_dict in zip(models[:5] + models[5:], weights,
                                             [input_dict_model3]*5 + [input_dict_model4]*5):
            probs = model.predict(input_dict, verbose=0)
            soft_preds = probs * weight if soft_preds is None else soft_preds + probs * weight


#UI Tier Shift from 0 to 4 to 1 to 5
#ONLY in frontend display.
#Displays explanatory business interpretation of tiers.


        final_pred = np.argmax(soft_preds, axis=1)[0]

        # ----------------------------------------
        # 11. Display Result
        # ----------------------------------------
        tier_map = {
            0: 'Minimal viewership',
            1: 'Low viewership',
            2: 'Average viewership',
            3: 'High viewership',
            4: 'Max viewership'
        }

        st.success(f"Predicted Tier: {final_pred + 1} - {tier_map[final_pred]}")

        # Note with Markdown formatting

        note = (
        "The predicted value tier is determined based on a five-point scale, ranging from lowest (Tier:1) to highest (Tier:5). "
        "The tiers are categorized as follows:\n\n"
        
        "• **Predicted Tier: 1 - Minimal Viewership**: Less than 112 TVTs  \n"
        "• **Predicted Tier: 2 - Low Viewership**: 112 to 144 TVTs  \n"
        "• **Predicted Tier: 3 - Average Viewership**: 144 to 195 TVTs  \n"
        "• **Predicted Tier: 4 - High Viewership**: 195 to 272 TVTs  \n"
        "• **Predicted Tier: 5 - Maximum Viewership**: 272 TVTs and above."
)
        st.markdown(note)

#If authentication fails:shows invalid credentials message.
#If nothing entered, asks user to log in.
# -------------------------------
# 12. Login Failure Handling
# -------------------------------
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')

# -------------------------------
# 13. Footer
# -------------------------------
st.write("""
---
**Note**: This app leverages Artificial Intelligence (AI) to predict news ratings, offering insights based on historical data. 
Predictions should be combined with domain expertise. The developer is not responsible for outcomes based solely on the app's predictions. 
For technical details on ML models employed and error metrics, contact: 
**Puneet Sah**  
📧 puneet.sah@timesgroup.com  
📞 9820615085
""")
