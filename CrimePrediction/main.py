import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, FastMarkerCluster
from sklearn.preprocessing import LabelEncoder
import os
import plotly.express as px

EARTH_RADIUS_MILES = 3958.8
TOP_N_OVERALL = 5
USER_LAT = 47.6062
USER_LON = -122.3321

st.set_page_config(
    page_title="Crime Data Analysis and Prediction",
    layout="wide"
)
st.title("📊 Crime Data Analysis and Prediction App")


@st.cache_data(ttl=3600, show_spinner=False)
def load_local_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['offense_start_datetime'])
        st.success("✅ Data loaded successfully from the local dataset.")
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found at path: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ An error occurred while loading data: {e}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        classifier_risk = joblib.load('models/classifier_risk.pkl')
        classifier_crime = joblib.load('models/classifier_crime.pkl')
        le_crime = joblib.load('models/le_crime.pkl')
        le_risk = joblib.load('models/le_risk.pkl')
        st.success("✅ Models and encoders loaded successfully.")
        return classifier_risk, classifier_crime, le_crime, le_risk
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None


def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_MILES * c


@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_data(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('/', '_')
    )

    needed_cols = [
        'offense_start_datetime', 'precinct', 'offense_parent_group',
        'latitude', 'longitude', 'crime_against_category'
    ]

    missing_cols = set(needed_cols) - set(df.columns)
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        return pd.DataFrame(), None, None

    df = df[needed_cols].dropna()
    df['offense_parent_group'] = df['offense_parent_group'].str.strip().str.upper()
    df[['precinct', 'offense_parent_group', 'crime_against_category']] = df[
        ['precinct', 'offense_parent_group', 'crime_against_category']].astype('category')

    max_date = df['offense_start_datetime'].max()
    df = df[df['offense_start_datetime'] >= (max_date - pd.DateOffset(years=3))]

    offense_counts = df['offense_parent_group'].value_counts()
    rare_offenses = offense_counts[offense_counts <= 1].index
    if len(rare_offenses) > 0:
        st.info(f"ℹ️ Removing rare offenses: {rare_offenses.tolist()}")
        df = df[~df['offense_parent_group'].isin(rare_offenses)]

    high_risk_categories = [
        'ASSAULT_OFFENSES', 'HOMICIDE_OFFENSES', 'KIDNAPPING_ABDUCTION', 'ROBBERY',
        'WEAPON_LAW_VIOLATIONS', 'BURGLARY_BREAKING&ENTERING', 'MOTOR_VEHICLE_THEFT',
        'DESTRUCTION_DAMAGE_VANDALISM_OF_PROPERTY', 'DRUG_NARCOTIC_OFFENSES',
        'EXTORTION_BLACKMAIL', 'HUMAN_TRAFFICKING'
    ]
    df['risk_level'] = np.where(df['offense_parent_group'].isin(high_risk_categories), 'High', 'Low')

    le_crime = LabelEncoder()
    le_risk = LabelEncoder()
    df['Crime_Type'] = le_crime.fit_transform(df['offense_parent_group'])
    df['risk_level_encoded'] = le_risk.fit_transform(df['risk_level'])

    df['month'] = df['offense_start_datetime'].dt.month
    df['day_of_week'] = df['offense_start_datetime'].dt.dayofweek
    df['hour'] = df['offense_start_datetime'].dt.hour

    df = pd.get_dummies(df, columns=['month', 'day_of_week', 'hour'], drop_first=True)
    return df, le_crime, le_risk


@st.cache_data(ttl=3600, show_spinner=False)
def prepare_features(df):
    feature_cols = [
        'latitude', 'longitude',
        'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
        'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
        'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4',
        'day_of_week_5', 'day_of_week_6',
        'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
        'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
        'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
        'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23'
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols].copy()


with st.spinner('🔄 Loading and processing data...'):
    local_data_path = './filtered_dataset_2021_to_now.csv'

    if not os.path.exists(local_data_path):
        st.error(f"❌ The dataset file was not found at: {local_data_path}")
        st.stop()

    df_raw = load_local_data(local_data_path)
    if df_raw.empty:
        st.warning("⚠️ No data loaded.")
        st.stop()

    df_preprocessed, le_crime, le_risk = preprocess_data(df_raw)
    if df_preprocessed.empty:
        st.warning("⚠️ Preprocessed data is empty.")
        st.stop()

    X = prepare_features(df_preprocessed)
    y_risk = df_preprocessed['risk_level_encoded']
    y_crime = df_preprocessed['Crime_Type']

    classifier_risk, classifier_crime, le_crime_loaded, le_risk_loaded = load_models()
    if not all([classifier_risk, classifier_crime, le_crime_loaded, le_risk_loaded]):
        st.error("❌ Model or encoders failed to load.")
        st.stop()

st.sidebar.header("🔍 Filter and View Crime Data")
st.sidebar.subheader("Select Time Range")

min_date = df_preprocessed['offense_start_datetime'].min().date()
max_date = df_preprocessed['offense_start_datetime'].max().date()

start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

if start_date > end_date:
    st.sidebar.error("❌ Start Date must be before End Date.")

st.sidebar.subheader("Select Crime Types")
crime_types = sorted(df_preprocessed['offense_parent_group'].unique())
selected_crime_types = st.sidebar.multiselect("Crime Types", options=crime_types, default=crime_types)

mask = (
        (df_preprocessed['offense_start_datetime'].dt.date >= start_date) &
        (df_preprocessed['offense_start_datetime'].dt.date <= end_date) &
        (df_preprocessed['offense_parent_group'].isin(selected_crime_types))
)
filtered_df = df_preprocessed[mask].copy()

st.subheader("📍 Crime Incidents Map")
st.markdown(f"📅 Displaying data from **{start_date}** to **{end_date}**")
st.markdown("""
**Interact with the Map**: Click on any location on the map to view heatmaps of crimes within a 1-mile radius 
for three distinct periods (2021-2022, 2022-2023, and 2023-2024) and see detailed risk assessments.
""")

m = folium.Map(location=[USER_LAT, USER_LON], zoom_start=13)
FastMarkerCluster(data=filtered_df[['latitude', 'longitude']].values.tolist()).add_to(m)
map_data = st_folium(m, width=700, height=500)

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    clicked_location = (clicked["lat"], clicked["lng"])
    st.write(f"📍 **Clicked Location:** Latitude {clicked_location[0]:.4f}, Longitude {clicked_location[1]:.4f}")

    nearby_df = df_preprocessed[
        haversine_vectorized(
            df_preprocessed['latitude'].values,
            df_preprocessed['longitude'].values,
            clicked_location[0],
            clicked_location[1]
        ) <= 1
        ].copy()

    if not nearby_df.empty:
        max_date_nearby = nearby_df['offense_start_datetime'].max()
        period_1_start = max_date_nearby - pd.DateOffset(years=1)  # 2023-2024
        period_2_start = max_date_nearby - pd.DateOffset(years=2)  # 2022-2023
        period_3_start = max_date_nearby - pd.DateOffset(years=3)  # 2021-2022

        nearby_df_recent = nearby_df[(nearby_df['offense_start_datetime'] > period_1_start) & (
                    nearby_df['offense_start_datetime'] <= max_date_nearby)]
        nearby_df_previous = nearby_df[(nearby_df['offense_start_datetime'] > period_2_start) & (
                    nearby_df['offense_start_datetime'] <= period_1_start)]
        nearby_df_old = nearby_df[(nearby_df['offense_start_datetime'] > period_3_start) & (
                    nearby_df['offense_start_datetime'] <= period_2_start)]


        def add_heatmap(f_map, data, gradient, name):
            if not data.empty:
                heat_data = data[['latitude', 'longitude']].values.tolist()
                HeatMap(
                    heat_data,
                    radius=15,
                    blur=10,
                    max_zoom=13,
                    name=name,
                    gradient=gradient,
                    min_opacity=0.2,
                    max_opacity=0.8
                ).add_to(f_map)


        heat_map = folium.Map(location=clicked_location, zoom_start=14)
        add_heatmap(heat_map, nearby_df_recent, {0.2: 'blue', 0.4: 'lime', 0.6: 'red'}, '2023-2024 Crimes')
        add_heatmap(heat_map, nearby_df_previous, {0.2: 'purple', 0.4: 'orange', 0.6: 'yellow'}, '2022-2023 Crimes')
        add_heatmap(heat_map, nearby_df_old, {0.2: 'pink', 0.4: 'cyan', 0.6: 'magenta'}, '2021-2022 Crimes')

        folium.Circle(location=clicked_location, radius=1609, color='green', fill=False).add_to(heat_map)
        folium.LayerControl().add_to(heat_map)

        st.subheader("🔥 Heatmap of Crimes within 1 Mile")
        st_folium(heat_map, width=700, height=500)


        def compute_risk_level(df_period):
            total = len(df_period)
            high_risk = (df_period['risk_level'] == 'High').sum()
            low_risk = total - high_risk
            risk_level = 'High' if (total > 0 and (high_risk / total) >= 0.5) else 'Low'
            return total, high_risk, low_risk, risk_level


        total_recent, high_risk_recent, low_risk_recent, risk_level_recent = compute_risk_level(nearby_df_recent)
        total_previous, high_risk_previous, low_risk_previous, risk_level_previous = compute_risk_level(
            nearby_df_previous)
        total_old, high_risk_old, low_risk_old, risk_level_old = compute_risk_level(nearby_df_old)

        st.markdown(f"### **Risk Level (2023-2024):** {risk_level_recent}")
        st.markdown(f"- Total incidents: {total_recent}")
        st.markdown(f"- High Risk: {high_risk_recent} ({high_risk_recent / total_recent:.2%})")
        st.markdown(f"- Low Risk: {low_risk_recent} ({low_risk_recent / total_recent:.2%})")

        st.markdown(f"### **Risk Level (2022-2023):** {risk_level_previous}")
        st.markdown(f"- Total incidents: {total_previous}")
        st.markdown(f"- High Risk: {high_risk_previous} ({high_risk_previous / total_previous:.2%})")
        st.markdown(f"- Low Risk: {low_risk_previous} ({low_risk_previous / total_previous:.2%})")

        st.markdown(f"### **Risk Level (2021-2022):** {risk_level_old}")
        st.markdown(f"- Total incidents: {total_old}")
        st.markdown(f"- High Risk: {high_risk_old} ({high_risk_old / total_old:.2%})")
        st.markdown(f"- Low Risk: {low_risk_old} ({low_risk_old / total_old:.2%})")

        st.markdown("""
        **Risk Level Explanations:**
        - **High Risk**: ≥50% incidents are high-risk (e.g., assault, robbery).
        - **Low Risk**: <50% incidents are high-risk.
        """)

        combined_counts = (
                nearby_df_recent['offense_parent_group'].value_counts() +
                nearby_df_previous['offense_parent_group'].value_counts() +
                nearby_df_old['offense_parent_group'].value_counts()
        ).fillna(0).sort_values(ascending=False)

        top_crime_types_vicinity = combined_counts.head(TOP_N_OVERALL).index.tolist()

        # Filter dataframes to top 5 crimes
        nearby_df_recent_top = nearby_df_recent[nearby_df_recent['offense_parent_group'].isin(top_crime_types_vicinity)]
        nearby_df_previous_top = nearby_df_previous[
            nearby_df_previous['offense_parent_group'].isin(top_crime_types_vicinity)]
        nearby_df_old_top = nearby_df_old[nearby_df_old['offense_parent_group'].isin(top_crime_types_vicinity)]


        def get_counts(df, period):
            counts = df['offense_parent_group'].value_counts().rename_axis('Crime_Type').reset_index(name='Count')
            counts['Period'] = period
            return counts


        crime_recent_counts = get_counts(nearby_df_recent_top, '2023-2024')
        crime_previous_counts = get_counts(nearby_df_previous_top, '2022-2023')
        crime_old_counts = get_counts(nearby_df_old_top, '2021-2022')

        crime_comparison = pd.concat([crime_recent_counts, crime_previous_counts, crime_old_counts])

        crime_comparison_pivot = (
            crime_comparison
            .pivot(index='Crime_Type', columns='Period', values='Count')
            .fillna(0)
            .reset_index()
        )

        crime_comparison_pivot = crime_comparison_pivot[
            crime_comparison_pivot['Crime_Type'].isin(top_crime_types_vicinity)]

        # Sort by total incidents and ensure only top 5
        crime_comparison_pivot['Total'] = crime_comparison_pivot[['2021-2022', '2022-2023', '2023-2024']].sum(axis=1)
        crime_comparison_pivot = crime_comparison_pivot.sort_values(by='Total', ascending=False).head(TOP_N_OVERALL)

        crime_comparison_pivot.rename(columns={'Crime_Type': 'Crime Type'}, inplace=True)

        st.markdown(f"### **Top {TOP_N_OVERALL} Significant Crime Types (Vicinity Comparison)**")
        fig_crime_comparison = px.bar(
            crime_comparison_pivot,
            x='Crime Type',
            y=['2021-2022', '2022-2023', '2023-2024'],
            labels={'value': 'Number of Incidents', 'Crime Type': 'Crime Type'},
            title=f'Top {TOP_N_OVERALL} Crime Types Over Three Periods (Vicinity)',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_crime_comparison.update_layout(
            xaxis_tickangle=-45,
            yaxis_title='Number of Incidents',
            legend_title='Period',
            title_x=0.5
        )
        fig_crime_comparison.update_traces(texttemplate='%{y}', textposition='auto')
        st.plotly_chart(fig_crime_comparison, use_container_width=True)
    else:
        st.info("ℹ️ Click on the map to view detailed crime heatmaps and assessments.")

st.subheader("🔍 Overall Crime Analysis")

# Trend Analysis
df_trend = df_preprocessed.copy()
df_trend['year_month'] = df_trend['offense_start_datetime'].dt.to_period('M').astype(str)
trend_data = df_trend.groupby('year_month').size().reset_index(name='Incident Count')
trend_data['year_month'] = pd.to_datetime(trend_data['year_month'])

min_date_dt = df_preprocessed['offense_start_datetime'].min()
max_date_dt = df_preprocessed['offense_start_datetime'].max()
trend_data['year'] = trend_data['year_month'].dt.year
trend_data['month'] = trend_data['year_month'].dt.month
trend_data['days_in_month'] = trend_data['year_month'].dt.days_in_month


def compute_days_in_data(row):
    period = row['year_month'].to_period('M')
    if period == min_date_dt.to_period('M'):
        return (min_date_dt + pd.offsets.MonthEnd(0)).day - min_date_dt.day + 1
    elif period == max_date_dt.to_period('M'):
        return max_date_dt.day
    return row['days_in_month']


trend_data['days_in_data'] = trend_data.apply(compute_days_in_data, axis=1)
trend_data['Adjusted Incident Count'] = trend_data['Incident Count'] / trend_data['days_in_data'] * trend_data[
    'days_in_month']

st.markdown("#### 📈 Crime Trends Over Time")
fig_trend = px.line(
    trend_data,
    x='year_month',
    y='Adjusted Incident Count',
    title='Monthly Crime Incidents Over Time',
    labels={'year_month': 'Month', 'Adjusted Incident Count': 'Number of Incidents'},
    markers=True
)
fig_trend.update_layout(xaxis=dict(tickformat='%b %Y'))
st.plotly_chart(fig_trend, use_container_width=True)

# Overall top crime types comparison
current_period_overall = df_preprocessed[
    df_preprocessed['offense_start_datetime'] >= (max_date_dt - pd.DateOffset(years=1))]
previous_period_overall = df_preprocessed[
    (df_preprocessed['offense_start_datetime'] >= (max_date_dt - pd.DateOffset(years=2))) &
    (df_preprocessed['offense_start_datetime'] < (max_date_dt - pd.DateOffset(years=1)))
    ]
old_period_overall = df_preprocessed[
    (df_preprocessed['offense_start_datetime'] >= (max_date_dt - pd.DateOffset(years=3))) &
    (df_preprocessed['offense_start_datetime'] < (max_date_dt - pd.DateOffset(years=2)))
    ]

crime_current_overall_counts = current_period_overall['offense_parent_group'].value_counts().rename_axis(
    'Crime_Type').reset_index(name='Count')
crime_previous_overall_counts = previous_period_overall['offense_parent_group'].value_counts().rename_axis(
    'Crime_Type').reset_index(name='Count')
crime_old_overall_counts = old_period_overall['offense_parent_group'].value_counts().rename_axis(
    'Crime_Type').reset_index(name='Count')

top_crime_types_overall = crime_current_overall_counts.head(TOP_N_OVERALL)['Crime_Type'].tolist()


def filter_top_periods(df):
    return df[df['Crime_Type'].isin(top_crime_types_overall)].set_index('Crime_Type').reindex(top_crime_types_overall,
                                                                                              fill_value=0).reset_index()


crime_current_overall_counts = filter_top_periods(crime_current_overall_counts)
crime_previous_overall_counts = filter_top_periods(crime_previous_overall_counts)
crime_old_overall_counts = filter_top_periods(crime_old_overall_counts)

crime_current_overall_counts['Period'] = '2023-2024'
crime_previous_overall_counts['Period'] = '2022-2023'
crime_old_overall_counts['Period'] = '2021-2022'

crime_comparison_overall = pd.concat(
    [crime_current_overall_counts, crime_previous_overall_counts, crime_old_overall_counts])
crime_comparison_pivot_overall = crime_comparison_overall.pivot(index='Crime_Type', columns='Period',
                                                                values='Count').fillna(0).reset_index()
crime_comparison_pivot_overall.rename(columns={'Crime_Type': 'Crime Type'}, inplace=True)
crime_comparison_pivot_overall['Total'] = crime_comparison_pivot_overall[['2021-2022', '2022-2023', '2023-2024']].sum(
    axis=1)
crime_comparison_pivot_overall = crime_comparison_pivot_overall.sort_values(by='Total', ascending=False)

st.markdown(f"### Top {TOP_N_OVERALL} Significant Crime Types (Overall Comparison)")
fig_crime_comparison_overall = px.bar(
    crime_comparison_pivot_overall,
    x='Crime Type',
    y=['2021-2022', '2022-2023', '2023-2024'],
    labels={'value': 'Number of Incidents', 'Crime Type': 'Crime Type'},
    title=f'Top {TOP_N_OVERALL} Crime Types Over Three Periods (Overall)',
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set1
)
fig_crime_comparison_overall.update_layout(
    xaxis_tickangle=-45,
    yaxis_title='Number of Incidents',
    legend_title='Period',
    title_x=0.5
)
fig_crime_comparison_overall.update_traces(texttemplate='%{y}', textposition='auto')
st.plotly_chart(fig_crime_comparison_overall, use_container_width=True)
