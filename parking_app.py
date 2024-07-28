import streamlit as st
import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from catboost import Pool
import pickle

# Load models and data from pickle files
with open('models_and_data.pkl', 'rb') as f:
    data = pickle.load(f)

model_hbf = data['model_hbf']
train_df_hbf = data['train_df_hbf']
test_df_hbf = data['test_df_hbf']
std_dev_hbf = data['std_dev_hbf']
model_sb = data['model_sb']
train_df_sb = data['train_df_sb']
test_df_sb = data['test_df_sb']
std_dev_sb = data['std_dev_sb']
features = data['features']
categorical_features = data['categorical_features']
target = data['target']

# Streamlit app
st.title("Parking Space Availability Predictor")

# User inputs
st.markdown("### This CatBoost model is trained on data from 2016 to 2020")

location = st.selectbox("Enter location you are looking for parking", ["freiburgambahnhof", "freiburgschlossberg"])
input_date = st.date_input("Select a Date", value=pd.to_datetime("2024-01-01"))
input_time = st.time_input("Select a Time", value=pd.to_datetime("2024-01-01 00:00").time())
confidence_interval = st.selectbox("Select Confidence Interval", options=['90%', '95%', '99%'])

# Button to trigger prediction
if st.button('Predict'):
    # Combine date and time
    input_datetime = pd.to_datetime(f"{input_date} {input_time}")

    # Prepare input for prediction
    input_df = pd.DataFrame({
        'Day': [input_datetime.day],
        'Month': [input_datetime.month],
        'Hour': [input_datetime.hour],
        'Minute': [input_datetime.minute],
        'Hour-Minute': [f'{input_datetime.hour}.{input_datetime.minute}'],
        'DOW': [input_datetime.weekday()],
        'Weekend Flag': [input_datetime.weekday() >= 5],
        'Public Holiday': [input_datetime.date() in holidays.Germany(state='BW')]
    })

    if location == "freiburgambahnhof":
        # Model and test data for freiburgambahnhof
        model = model_hbf
        train_df = train_df_hbf
        test_df = test_df_hbf
        std_dev = std_dev_hbf
        max_parking_space = 243
    else:
        # Model and test data for freiburgschlossberg
        model = model_sb
        train_df = train_df_sb
        test_df = test_df_sb
        std_dev = std_dev_sb
        max_parking_space = 440

    # Make prediction
    input_pool = Pool(input_df[features], cat_features=categorical_features)
    predicted_value = model.predict(input_pool)[0]
    predicted_value_rounded = max(0, round(predicted_value))
    predicted_value_rounded = min(predicted_value_rounded, max_parking_space)

    # Calculate confidence interval
    confidence_levels = {'90%': 1.645, '95%': 1.96, '99%': 2.576}
    z = confidence_levels[confidence_interval]
    lower_limit = max(0, round(predicted_value_rounded - z * std_dev))
    upper_limit = min(max_parking_space, round(predicted_value_rounded + z * std_dev))

    # Calculate utilization rates
    predicted_utilization = round(((max_parking_space - predicted_value_rounded) / max_parking_space) * 100, 2)
    upper_utilization = round(((max_parking_space - upper_limit) / max_parking_space) * 100, 2)
    lower_utilization = round(((max_parking_space - lower_limit) / max_parking_space) * 100, 2)

    # Display prediction and utilization rate
    st.markdown(f"<p style='font-size: 18px;'>The number of predicted parking spots available is {predicted_value_rounded}. The predicted utilization rate of {location} is currently at {predicted_utilization}%.</p>", unsafe_allow_html=True)
    
    # Donut plot for predicted utilization rate
    fig1, ax1 = plt.subplots()
    sizes = [predicted_value_rounded, max_parking_space - predicted_value_rounded]
    colors = ['#ff9999','#66b3ff']
    ax1.pie(sizes, labels=['Available', 'Occupied'], autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig1.gca().add_artist(centre_circle)
    ax1.axis('equal')
    plt.title(f'{location.capitalize()} Parking Utilization')
    st.pyplot(fig1)

    st.markdown(f"<p style='font-size: 18px;'>With a confidence interval of {confidence_interval}, the highest possible parking spaces available could be {upper_limit} ({upper_utilization}% utilization rate) and the lowest possible parking spaces available could be {lower_limit} ({lower_utilization}% utilization rate).</p>", unsafe_allow_html=True)
    
    # Plot for highest and lowest possible parking spaces
    fig2, ax2 = plt.subplots()
    x_labels = ['Lower Limit', 'Predicted', 'Upper Limit']
    y_values = [lower_limit, predicted_value_rounded, upper_limit]
    utilization_values = [lower_utilization, predicted_utilization, upper_utilization]
    ax2.bar(x_labels, y_values, color=['#ff9999','#66b3ff', '#99ff99'])
    for i, v in enumerate(y_values):
        ax2.text(i, v + 2, f'{v}\n({utilization_values[i]}%)', ha='center')
    plt.ylim(0, max_parking_space + 20)
    plt.ylabel('Parking Spaces Available')
    plt.title('Predicted Parking Spaces with Confidence Interval')
    st.pyplot(fig2)
    
    st.markdown("<p style='font-size: 18px;'>This table shows the predicted available parking spaces in the next 1 hour from date and time entered</p>", unsafe_allow_html=True)
    
    time_slots = [input_datetime + pd.Timedelta(minutes=15 * i) for i in range(5)]
    table_data = []
    for time_slot in time_slots:
        time_slot_df = pd.DataFrame({
            'Day': [time_slot.day],
            'Month': [time_slot.month],
            'Hour': [time_slot.hour],
            'Minute': [time_slot.minute],
            'Hour-Minute': [f'{time_slot.hour}.{time_slot.minute}'],
            'DOW': [time_slot.weekday()],
            'Weekend Flag': [time_slot.weekday() >= 5],
            'Public Holiday': [time_slot.date() in holidays.Germany(state='BW')]
        })
        time_slot_pool = Pool(time_slot_df[features], cat_features=categorical_features)
        time_slot_prediction = model.predict(time_slot_pool)[0]
        time_slot_prediction_rounded = max(0, round(time_slot_prediction))
        time_slot_prediction_rounded = min(time_slot_prediction_rounded, max_parking_space)
        time_slot_upper_limit = min(max_parking_space, round(time_slot_prediction_rounded + z * std_dev))
        time_slot_lower_limit = max(0, round(time_slot_prediction_rounded - z * std_dev))
        time_slot_utilization = round(((max_parking_space - time_slot_prediction_rounded) / max_parking_space) * 100, 2)
        
        table_data.append({
            'Date': time_slot.date(),
            'Time': time_slot.time().strftime("%H:%M:%S"),
            'Predicted': time_slot_prediction_rounded,
            'Upper': time_slot_upper_limit,
            'Lower': time_slot_lower_limit,
            'UR%': time_slot_utilization
        })
    
    table_df = pd.DataFrame(table_data)
    st.table(table_df)

    # Find the best time slot for parking
    best_time_slot = table_df.loc[table_df['UR%'].idxmin()]
    st.markdown(f"<p style='font-size: 18px;'>The model says the best time to look for an available parking space at {location} in the next 1 hour is at {best_time_slot['Time']} which has {best_time_slot['UR%']}% utilization rate.</p>", unsafe_allow_html=True)

    # Compare with other location
    if location == "freiburgambahnhof":
        other_location = "freiburgschlossberg"
        other_model = model_sb
        other_test_df = test_df_sb
        other_max_parking_space = 440
    else:
        other_location = "freiburgambahnhof"
        other_model = model_hbf
        other_test_df = test_df_hbf
        other_max_parking_space = 243

    other_predicted_value = other_model.predict(input_pool)[0]
    other_predicted_value_rounded = max(0, round(other_predicted_value))
    other_predicted_value_rounded = min(other_predicted_value_rounded, other_max_parking_space)
    other_utilization = round(((other_max_parking_space - other_predicted_value_rounded) / other_max_parking_space) * 100, 2)

    if predicted_value_rounded > other_predicted_value_rounded:
        st.markdown(f"<p style='font-size: 18px;'>Between the 2 locations, you are more likely to find a parking at {location} because it has {predicted_value_rounded} predicted parking space available ({predicted_utilization}% utilization rate) than {other_location} with only {other_predicted_value_rounded} ({other_utilization}% utilization rate) predicted parking space available at the date time selected.</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-size: 18px;'>Between the 2 locations, you are more likely to find a parking at {other_location} because it has {other_predicted_value_rounded} predicted parking space available ({other_utilization}% utilization rate) than {location} with only {predicted_value_rounded} ({predicted_utilization}% utilization rate) predicted parking space available at the date time selected.</p>", unsafe_allow_html=True)
    
    # Display actual parking space available if known
    if input_datetime in test_df.index or input_datetime in train_df.index:
        if input_datetime in test_df.index:
            actual_value = test_df.loc[input_datetime, 'Parking Spaces Available']
        else:
            actual_value = train_df.loc[input_datetime, 'Parking Spaces Available']
        st.markdown(f"<p style='font-size: 18px;'>Since we know the actual parking space available for this date, it is: <b>{actual_value}</b></p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-size: 18px;'>This date and time does not exist in the data set so we do not know the actual parking slots available for this time slot.</p>", unsafe_allow_html=True)
    
    # Display feature importance plot
    shap_values = model.get_feature_importance(input_pool, type='ShapValues')
    shap_df = pd.DataFrame({
        'Feature': features,
        'Importance': shap_values.mean(axis=0)[:-1]
    }).sort_values(by='Importance', ascending=False)
    
    fig3, ax3 = plt.subplots()
    shap_df.plot(kind='barh', x='Feature', y='Importance', ax=ax3, legend=False)
    plt.xlabel('Importance')
    plt.title('Factors which influenced Prediction')
    for index, value in enumerate(shap_df['Importance']):
        ax3.text(value, index, f"{value:.2f}")
    st.pyplot(fig3)
