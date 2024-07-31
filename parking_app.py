import streamlit as st
import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from catboost import Pool
import pickle
import matplotlib.dates as mdates

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
st.markdown("##### This app, developed by Sharan, utilizes a CatBoost machine learning model trained on data from 2016 to 2020 to generate parking predictions. Please note that the model does not account for external factors such as weather conditions, construction work, public transport disruptions, or special events.")

location = st.selectbox("Enter the location where you are seeking parking", ["freiburgschlossberg","freiburgambahnhof"])
input_date = st.date_input("Select a Date", value=pd.to_datetime("2021-11-16"))
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
        'Public Holiday': [input_datetime.date() in holidays.Germany(state='BW')],
        'Year': [input_datetime.year]  # Added Year feature
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

    # Confidence intervals
    if confidence_interval == '90%':
        z = 1.645
    elif confidence_interval == '95%':
        z = 1.96
    else:  # '99%'
        z = 2.576

    upper_limit = round(min(max_parking_space, round(predicted_value_rounded + z * std_dev)),2)
    lower_limit = round(max(0, round(predicted_value_rounded - z * std_dev)),2)

    # Calculate utilization rates
    predicted_utilization = round(((max_parking_space - predicted_value_rounded) / max_parking_space) * 100, 2)
    predicted_availability = round((100 - predicted_utilization), 2)
    upper_utilization = round(((max_parking_space - upper_limit) / max_parking_space) * 100, 2)
    lower_utilization = round(((max_parking_space - lower_limit) / max_parking_space) * 100, 2)
    # Calculate availability rates
    upper_availability = round(100 - ((max_parking_space - upper_limit) / max_parking_space) * 100, 2)
    lower_availability = round(100 - ((max_parking_space - lower_limit) / max_parking_space) * 100, 2)
    
    st.markdown("##### 1. Predicted Parking Availability")
    # Display prediction and utilization rate
    st.markdown(f"""<p style='font-size: 18px;'>
    The predicted number of available parking spaces in {location} is {predicted_value_rounded}. 
    The probability of finding a parking space at this time is {predicted_availability}%. 
    The current predicted utilization rate for {location} is {predicted_utilization}%.</p>
    """, unsafe_allow_html=True)

    # Donut plot for predicted utilization rate
    fig1, ax1 = plt.subplots()
    sizes = [predicted_value_rounded, max_parking_space - predicted_value_rounded]
    colors = ['#99ff99', '#ff9999']
    ax1.pie(sizes, labels=['Available', 'Occupied'], autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig1.gca().add_artist(centre_circle)
    ax1.axis('equal')
    plt.title(f'{location.capitalize()} Parking Availability')
    st.pyplot(fig1)

    st.markdown(f"""<p style='font-size: 18px;'>With a confidence interval of {confidence_interval}, the number of available parking spaces could range from a maximum of {upper_limit} (availability rate of {upper_availability}%) to a minimum of {lower_limit} (availability rate of {lower_availability}%).</p>""", unsafe_allow_html=True)

    # Plot for highest and lowest possible parking spaces
    fig2, ax2 = plt.subplots()
    x_labels = ['Lower Limit', 'Predicted', 'Upper Limit']
    y_values = [lower_limit, predicted_value_rounded, upper_limit]
    availability_values = [lower_availability, predicted_availability, upper_availability]
    ax2.bar(x_labels, y_values, color=['#ff9999', '#66b3ff', '#99ff99'])
    for i, v in enumerate(y_values):
        ax2.text(i, v + 2, f'{v}\n({availability_values[i]}%)', ha='center')
    plt.ylim(0, max_parking_space + 20)
    plt.ylabel('Parking Spaces Available')
    plt.title('Predicted Parking Spaces with Confidence Interval')
    st.pyplot(fig2)

    st.markdown("##### 2. Optimal Time for Parking Availability")
    
    st.markdown("<p style='font-size: 18px;'>The table below displays the predicted available parking spaces for the next hour from the entered date and time.</p>", unsafe_allow_html=True)

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
            'Public Holiday': [time_slot.date() in holidays.Germany(state='BW')],
            'Year': [time_slot.year]  # Added Year feature
        })
        time_slot_pool = Pool(time_slot_df[features], cat_features=categorical_features)
        time_slot_prediction = model.predict(time_slot_pool)[0]
        time_slot_prediction_rounded = max(0, round(time_slot_prediction))
        time_slot_prediction_rounded = min(time_slot_prediction_rounded, max_parking_space)
        time_slot_upper_limit = min(max_parking_space, round(time_slot_prediction_rounded + z * std_dev))
        time_slot_lower_limit = max(0, round(time_slot_prediction_rounded - z * std_dev))
        time_slot_availability = round(((max_parking_space - time_slot_prediction_rounded) / max_parking_space) * 100, 2)

        table_data.append({
            'Date': time_slot.date(),
            'Time': time_slot.time().strftime("%H:%M:%S"),
            'Predicted': time_slot_prediction_rounded,
            'Upper': time_slot_upper_limit,
            'Lower': time_slot_lower_limit,
            'AR%': round(100 - time_slot_availability,2)
        })

    table_df = pd.DataFrame(table_data)
    st.table(table_df)

    # Find the best time slot for parking
    best_time_slot = table_df.loc[table_df['AR%'].idxmax()]
    st.markdown(f"<p style='font-size: 18px;'>According to the model, the best time to find an available parking space at {location} in the next hour is at {best_time_slot['Time']}, with an availability rate of {best_time_slot['AR%']}%.</p>", unsafe_allow_html=True)

    # Generate predictions for the entire day
    day_slots = [input_datetime.replace(hour=h, minute=0) + pd.Timedelta(minutes=15 * i) for h in range(24) for i in range(4)]
    day_data = []
    for slot in day_slots:
        slot_df = pd.DataFrame({
            'Day': [slot.day],
            'Month': [slot.month],
            'Hour': [slot.hour],
            'Minute': [slot.minute],
            'Hour-Minute': [f'{slot.hour}.{slot.minute}'],
            'DOW': [slot.weekday()],
            'Weekend Flag': [slot.weekday() >= 5],
            'Public Holiday': [slot.date() in holidays.Germany(state='BW')],
            'Year': [slot.year]  # Added Year feature
        })
        slot_pool = Pool(slot_df[features], cat_features=categorical_features)
        slot_prediction = model.predict(slot_pool)[0]
        slot_prediction_rounded = max(0, round(slot_prediction))
        slot_prediction_rounded = min(slot_prediction_rounded, max_parking_space)
        slot_upper_limit = min(max_parking_space, round(slot_prediction_rounded + z * std_dev))
        slot_lower_limit = max(0, round(slot_prediction_rounded - z * std_dev))

        day_data.append({
            'Time': slot,
            'Predicted': slot_prediction_rounded,
            'Upper': slot_upper_limit,
            'Lower': slot_lower_limit
        })

    day_df = pd.DataFrame(day_data)



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
    other_availability = round((100 - other_utilization), 2)

    st.markdown("##### 3. Comparision to other location")
    
    st.markdown(f"<p style='font-size: 18px;'>The model predicts {other_predicted_value_rounded} available parking spaces at {other_location} at the selected date and time, with an availability rate of {other_availability}%.</p>", unsafe_allow_html=True)

    if other_availability > predicted_availability:
        st.markdown(f"<p style='font-size: 18px;'>Between the two locations, you are more likely to find parking at {other_location}, which has {other_predicted_value_rounded} predicted available spaces (utilization rate of {other_utilization}%), compared to {location}, which has only {predicted_value_rounded} available spaces (utilization rate of {predicted_utilization}%) at the selected date and time.</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-size: 18px;'>Between the two locations, the model predicts a higher chance of finding an available parking space at {location} ({predicted_availability}%).</p>", unsafe_allow_html=True)
    
    
    # 4. Comparison to Actual Data
    st.markdown("##### 4. Comparison to Actual Data")
    
    # Display actual parking space available if known
    if input_datetime in test_df.index or input_datetime in train_df.index:
        if input_datetime in test_df.index:
            actual_value = test_df.loc[input_datetime, 'Parking Spaces Available']
        else:
            actual_value = train_df.loc[input_datetime, 'Parking Spaces Available']
        st.markdown(f"<p style='font-size: 18px;'>The actual number of available parking spaces for this date and time selected is: <b>{actual_value}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 18px;'>The predicted number of available parking spaces for this date and time selected is: <b>{predicted_value_rounded}</b></p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-size: 18px;'>This date and time are not present in the data set, so the actual number of available parking spaces for this time slot is unknown.</p>", unsafe_allow_html=True)
    
    # Generate predictions for the entire day
    day_slots = [input_datetime.replace(hour=h, minute=0) + pd.Timedelta(minutes=15 * i) for h in range(24) for i in range(4)]
    day_data = []
    actual_data = []
    
    for slot in day_slots:
        slot_df = pd.DataFrame({
            'Day': [slot.day],
            'Month': [slot.month],
            'Hour': [slot.hour],
            'Minute': [slot.minute],
            'Hour-Minute': [f'{slot.hour}.{slot.minute}'],
            'DOW': [slot.weekday()],
            'Weekend Flag': [slot.weekday() >= 5],
            'Public Holiday': [slot.date() in holidays.Germany(state='BW')],
            'Year': [slot.year]  # Added Year feature
        })
        slot_pool = Pool(slot_df[features], cat_features=categorical_features)
        slot_prediction = model.predict(slot_pool)[0]
        slot_prediction_rounded = max(0, round(slot_prediction))
        slot_prediction_rounded = min(slot_prediction_rounded, max_parking_space)
        slot_upper_limit = min(max_parking_space, round(slot_prediction_rounded + z * std_dev))
        slot_lower_limit = max(0, round(slot_prediction_rounded - z * std_dev))
    
        day_data.append({
            'Time': slot,
            'Predicted': slot_prediction_rounded,
            'Upper': slot_upper_limit,
            'Lower': slot_lower_limit
        })
    
        if slot in test_df.index:
            actual_value = test_df.loc[slot, 'Parking Spaces Available']
        elif slot in train_df.index:
            actual_value = train_df.loc[slot, 'Parking Spaces Available']
        else:
            actual_value = np.nan
    
        actual_data.append({
            'Time': slot,
            'Actual': actual_value
        })
    
    day_df = pd.DataFrame(day_data)
    actual_df = pd.DataFrame(actual_data)
    
    # Plot the predictions and actual data for the entire day
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(day_df['Time'], day_df['Predicted'], label='Predicted', color='blue')
    ax4.plot(day_df['Time'], day_df['Upper'], label='Upper Limit', color='green', linestyle='--')
    ax4.plot(day_df['Time'], day_df['Lower'], label='Lower Limit', color='red', linestyle='--')
    ax4.fill_between(day_df['Time'], day_df['Lower'], day_df['Upper'], color='gray', alpha=0.2)
    ax4.plot(actual_df['Time'], actual_df['Actual'], label='Actual', color='orange', linestyle='-', marker='o')
    
    # Format the x-axis to display time properly
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel('Time')
    plt.ylabel('Parking Slots Available')
    plt.title(f'Parking Slots Availability Prediction vs. Actual for {input_date}')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig4)
    
    st.markdown("##### 5. Factors Influencing Prediction")
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

    st.markdown("<p style='font-size: 18px;'>The above bar chart displays the factors that influenced the prediction. The feature importance is calculated using SHAP values, which indicate the impact of each feature on the prediction.</p>", unsafe_allow_html=True)
    
    # Compute the absolute values and find the feature with the highest absolute SHAP value
    shap_df['Absolute Importance'] = shap_df['Importance'].abs()
    max_abs_feature = shap_df.loc[shap_df['Absolute Importance'].idxmax()]
    
    # Extract the feature with the highest absolute SHAP value
    feature = max_abs_feature['Feature']
    importance_value = max_abs_feature['Importance']
    
    # Generate explanation based on the sign of the SHAP value
    if importance_value > 0:
        explanation = (
            f"<p style='font-size: 18px;'>{feature} has the highest impact on the prediction with a SHAP value of {importance_value:.2f}. "
            f"A high positive value for {feature} suggests that there is generally better parking availability at this {feature.lower()} "
            f"and hence this feature suggests the model to increase the prediction.</p>"
        )
    else:
        explanation = (
            f"<p style='font-size: 18px;'>{feature} has the highest impact on the prediction with a SHAP value of {importance_value:.2f}. "
            f"A high negative value for {feature} suggests that there is generally low parking availability at this {feature.lower()} "
            f"and hence this feature suggests the model to decrease the prediction.</p>"
        )
    
    # Display the generated explanation text
    st.markdown(explanation, unsafe_allow_html=True)
