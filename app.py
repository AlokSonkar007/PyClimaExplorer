import streamlit as st
import xarray as xr
import tempfile
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="PyClimaExplorer", page_icon="", layout="wide")

# Path to bundled sample dataset — loaded automatically on first visit
# Streamlit Cloud always runs with CWD = repo root, so a plain filename works.
# For local use, we also try the script's own directory as a fallback.
_SAMPLE_FILENAME = "air.sig995.2020.nc"
if os.path.exists(_SAMPLE_FILENAME):
    SAMPLE_DATA_PATH = _SAMPLE_FILENAME
else:
    try:
        SAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _SAMPLE_FILENAME)
    except Exception:
        SAMPLE_DATA_PATH = _SAMPLE_FILENAME

st.markdown("""
<style>
    .stApp {
        background-color: #F5F5DC;
    }
    .main-title {font-size:2.2rem;font-weight:700;margin-bottom:0.2rem;color:#333;}
    .sub-title {font-size:1rem;color:#666;margin-bottom:1.5rem;}
    .insight-card {
        background: #1e1e2f;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        color: #f0f0f0;
        border-left: 4px solid #636EFA;
    }
    .disaster-card {
        background: #2b1111;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #f0f0f0;
        border-left: 4px solid #e63946;
    }
    .disaster-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e63946;
        margin-bottom: 0.5rem;
    }
    .live-card {
        background: #0a2e1a;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #f0f0f0;
        border-left: 4px solid #2ecc71;
    }
    .live-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2ecc71;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">PyClimaExplorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Climate Data Dashboard</div>', unsafe_allow_html=True)

DISASTERS = [
    {
        "name": "1997-98 El Nino",
        "time": "1998-01-01",
        "lat": 0.0,
        "lon": 180.0,
        "category": "Ocean Warming",
        "impact": "One of the strongest El Nino events. Caused extreme flooding in South America, droughts in Southeast Asia, coral bleaching worldwide. Estimated $35 billion in damages globally."
    },
    {
        "name": "2003 European Heat Wave",
        "time": "2003-08-01",
        "lat": 46.0,
        "lon": 2.0,
        "category": "Extreme Heat",
        "impact": "Temperatures exceeded 40C across Europe for weeks. Over 70,000 excess deaths recorded. France, Germany, and Italy were hardest hit."
    },
    {
        "name": "2010 Russian Heat Wave",
        "time": "2010-07-01",
        "lat": 55.0,
        "lon": 40.0,
        "category": "Extreme Heat",
        "impact": "Record temperatures of 44C in parts of Russia. Caused massive wildfires, crop failures, and an estimated 56,000 excess deaths."
    },
    {
        "name": "2015-16 El Nino",
        "time": "2016-01-01",
        "lat": 0.0,
        "lon": 160.0,
        "category": "Ocean Warming",
        "impact": "Rivaled 1997-98 in strength. 2016 became the hottest year on record. Massive coral bleaching on the Great Barrier Reef."
    },
    {
        "name": "2020 Arctic Heat",
        "time": "2020-06-01",
        "lat": 70.0,
        "lon": 100.0,
        "category": "Polar Anomaly",
        "impact": "Siberia hit 38C in Verkhoyansk -- highest ever recorded above the Arctic Circle. Triggered massive wildfires and permafrost thaw."
    },
    {
        "name": "2023 Global Heat Records",
        "time": "2023-07-01",
        "lat": 35.0,
        "lon": -10.0,
        "category": "Global Warming",
        "impact": "2023 confirmed as hottest year in recorded history. July 2023 was the hottest month ever. Ocean temperatures hit unprecedented levels."
    }
]

LOCATIONS = {
    "-- Custom (use sliders) --": (None, None),
    "New Delhi, India": (28.6, 77.2),
    "Mumbai, India": (19.1, 72.9),
    "New York, USA": (40.7, -74.0),
    "London, UK": (51.5, -0.1),
    "Tokyo, Japan": (35.7, 139.7),
    "Sydney, Australia": (-33.9, 151.2),
    "Cairo, Egypt": (30.0, 31.2),
    "Moscow, Russia": (55.8, 37.6),
    "Sao Paulo, Brazil": (-23.5, -46.6),
    "Beijing, China": (39.9, 116.4),
    "Nairobi, Kenya": (-1.3, 36.8),
    "Arctic (North Pole)": (85.0, 0.0),
    "Antarctic (South Pole)": (-85.0, 0.0),
    "Central Pacific (El Nino)": (0.0, 180.0),
    "Sahara Desert": (23.0, 12.0),
    "Amazon Rainforest": (-3.0, -60.0),
}

def fetch_live_events():
    try:
        url = "https://eonet.gsfc.nasa.gov/api/v3/events?limit=30&status=open"
        response = requests.get(url, timeout=10)
        data = response.json()
        events = []
        for event in data.get("events", []):
            title = event.get("title", "Unknown")
            category = event.get("categories", [{}])[0].get("title", "Unknown")
            geometry = event.get("geometry", [])
            if len(geometry) == 0:
                continue
            latest = geometry[-1]
            coords = latest.get("coordinates", [0, 0])
            date = latest.get("date", "")[:10]
            events.append({
                "name": title,
                "time": date,
                "lat": float(coords[1]),
                "lon": float(coords[0]),
                "category": category,
                "impact": f"Live event detected by NASA EONET. Category: {category}. Location: ({coords[1]:.1f}, {coords[0]:.1f}). Last updated: {date}."
            })
        return events
    except:
        return []

uploaded_file = st.file_uploader("Upload a NetCDF file", type=["nc"])

# Determine which dataset to load: uploaded file takes priority, otherwise fall back to sample data
using_sample = uploaded_file is None
if using_sample:
    if os.path.exists(SAMPLE_DATA_PATH):
        ds = xr.open_dataset(SAMPLE_DATA_PATH)
        ds.load()
        ds.close()
        st.info("📊 Showing sample dataset: **air.sig995.2020.nc** (Near-Surface Air Temperature, 2020). Upload your own .nc file above to explore your data.")
    else:
        st.error(f"⚠️ Sample file not found at: `{os.path.abspath(SAMPLE_DATA_PATH)}` — CWD is `{os.getcwd()}`. Please upload a .nc file manually.")
        st.stop()
else:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    ds = xr.open_dataset(tmp.name)
    ds.load()
    ds.close()
    os.unlink(tmp.name)

if True:  # dashboard always runs once ds is loaded

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables", len(list(ds.data_vars)))
    col2.metric("Dimensions", len(ds.dims))
    col3.metric("Total Size (MB)", round(ds.nbytes / 1e6, 2))

    st.markdown("---")

    control_col1, control_col2 = st.columns(2)

    all_vars = list(ds.data_vars)
    selected_var = control_col1.selectbox("Select Variable", all_vars)

    time_dim = None
    for d in ds.dims:
        if d.lower() in ["time", "t", "date", "datetime"]:
            time_dim = d
            break

    if time_dim is not None:
        time_values = pd.to_datetime(ds.coords[time_dim].values)
        time_min = time_values.min().to_pydatetime()
        time_max = time_values.max().to_pydatetime()

        selected_time = control_col2.date_input(
            "Select Time Range",
            value=(time_min.date(), time_max.date()),
            min_value=time_min.date(),
            max_value=time_max.date()
        )

        if len(selected_time) == 2:
            selected_time = (pd.Timestamp(selected_time[0]), pd.Timestamp(selected_time[1]))
        else:
            selected_time = (pd.Timestamp(selected_time[0]), pd.Timestamp(time_max))

        ds_filtered = ds[selected_var].sel({time_dim: slice(selected_time[0], selected_time[1])})
    else:
        control_col2.warning("No time dimension found.")
        ds_filtered = ds[selected_var]

    st.markdown("---")

    lat_name = None
    lon_name = None
    for d in ds_filtered.dims:
        if d.lower() in ["lat", "latitude"]:
            lat_name = d
        if d.lower() in ["lon", "longitude"]:
            lon_name = d

    st.subheader("Spatial View")

    if lat_name and lon_name:
        if time_dim is not None and time_dim in ds_filtered.dims:
            time_steps = pd.to_datetime(ds_filtered.coords[time_dim].values)
            picked_time = st.select_slider("Pick a time slice for map", options=time_steps, value=time_steps[0])
            slice_data = ds_filtered.sel({time_dim: picked_time})
        else:
            slice_data = ds_filtered

        lat_vals = slice_data.coords[lat_name].values
        lon_vals = slice_data.coords[lon_name].values
        z_vals = slice_data.values

        if z_vals.ndim != 2:
            remaining_dims = [d for d in slice_data.dims if d != lat_name and d != lon_name]
            for rd in remaining_dims:
                slice_data = slice_data.isel({rd: 0})
            z_vals = slice_data.values

        fig = px.imshow(
            z_vals,
            x=lon_vals,
            y=lat_vals,
            color_continuous_scale="RdBu_r",
            labels={"x": "Longitude", "y": "Latitude", "color": selected_var},
            aspect="auto",
            origin="lower"
        )

        fig.update_layout(
            height=500,
            margin=dict(l=60, r=60, t=30, b=20)
        )

        pad1, spatial_main, pad2 = st.columns([0.3, 5, 0.3])
        with spatial_main:
            st.plotly_chart(fig)

        st.markdown("---")

        st.subheader("Animated Timelapse")
        st.caption("Watch how the data changes over time")

        if time_dim is not None and time_dim in ds_filtered.dims:
            n_frames = min(20, len(time_steps))
            frame_indices = np.linspace(0, len(time_steps) - 1, n_frames, dtype=int)

            step_lat = max(1, len(lat_vals) // 60)
            step_lon = max(1, len(lon_vals) // 60)
            lat_anim = lat_vals[::step_lat]
            lon_anim = lon_vals[::step_lon]

            all_frames_data = []
            for idx in frame_indices:
                t = time_steps[idx]
                frame_slice = ds_filtered.sel({time_dim: t})
                fz = frame_slice.values
                if fz.ndim != 2:
                    for rd in [dd for dd in frame_slice.dims if dd != lat_name and dd != lon_name]:
                        frame_slice = frame_slice.isel({rd: 0})
                    fz = frame_slice.values
                all_frames_data.append(fz[::step_lat, ::step_lon])

            global_zmin = min(np.nanmin(f) for f in all_frames_data)
            global_zmax = max(np.nanmax(f) for f in all_frames_data)

            fig_anim = go.Figure(
                data=go.Heatmap(
                    z=all_frames_data[0],
                    x=lon_anim,
                    y=lat_anim,
                    colorscale="RdBu_r",
                    zmin=global_zmin,
                    zmax=global_zmax,
                    colorbar=dict(title=selected_var)
                )
            )

            frames = []
            for i, idx in enumerate(frame_indices):
                frames.append(go.Frame(
                    data=[go.Heatmap(
                        z=all_frames_data[i],
                        x=lon_anim,
                        y=lat_anim,
                        colorscale="RdBu_r",
                        zmin=global_zmin,
                        zmax=global_zmax
                    )],
                    name=str(time_steps[idx].strftime("%Y-%m-%d"))
                ))

            fig_anim.frames = frames

            fig_anim.update_layout(
                height=500,
                margin=dict(l=60, r=60, t=30, b=20),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    x=0.0,
                    y=-0.05,
                    buttons=[
                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                        dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                    ]
                )],
                sliders=[dict(
                    active=0,
                    steps=[dict(
                        args=[[str(time_steps[frame_indices[i]].strftime("%Y-%m-%d"))], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=str(time_steps[frame_indices[i]].strftime("%Y-%m-%d")),
                        method="animate"
                    ) for i in range(n_frames)],
                    x=0.0,
                    len=1.0,
                    y=-0.1
                )]
            )

            pad3, anim_main, pad4 = st.columns([0.3, 5, 0.3])
            with anim_main:
                st.plotly_chart(fig_anim)

        st.markdown("---")

        st.subheader("Temporal View")

        lat_list = sorted(lat_vals.tolist())
        lon_list = sorted(lon_vals.tolist())

        location_pick = st.selectbox("Quick Location Jump", list(LOCATIONS.keys()))

        if LOCATIONS[location_pick][0] is not None:
            target_lat = LOCATIONS[location_pick][0]
            target_lon = LOCATIONS[location_pick][1]
            default_lat = min(lat_list, key=lambda x: abs(x - target_lat))
            default_lon = min(lon_list, key=lambda x: abs(x - target_lon))
        else:
            default_lat = lat_list[len(lat_list) // 2]
            default_lon = lon_list[len(lon_list) // 2]

        loc_col1, loc_col2 = st.columns(2)

        picked_lat = loc_col1.select_slider("Select Latitude", options=lat_list, value=default_lat)
        picked_lon = loc_col2.select_slider("Select Longitude", options=lon_list, value=default_lon)

        if time_dim is not None and time_dim in ds_filtered.dims:
            point_data = ds_filtered.sel(
                {lat_name: picked_lat, lon_name: picked_lon},
                method="nearest"
            )

            extra_dims = [d for d in point_data.dims if d != time_dim]
            for ed in extra_dims:
                point_data = point_data.isel({ed: 0})

            time_axis = pd.to_datetime(point_data.coords[time_dim].values)
            val_axis = point_data.values

            mean_val = np.nanmean(val_axis)
            std_val = np.nanstd(val_axis)

            if std_val > 0:
                z_scores = (val_axis - mean_val) / std_val
            else:
                z_scores = np.zeros_like(val_axis)

            anomaly_mask = np.abs(z_scores) > 2
            anomaly_times = time_axis[anomaly_mask]
            anomaly_vals = val_axis[anomaly_mask]

            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=time_axis,
                y=val_axis,
                mode="lines",
                name=selected_var,
                line=dict(color="#636EFA")
            ))

            window = min(12, len(val_axis) // 4)
            if window > 1:
                moving_avg = pd.Series(val_axis).rolling(window=window, center=True).mean()
                fig2.add_trace(go.Scatter(
                    x=time_axis,
                    y=moving_avg,
                    mode="lines",
                    name=f"Moving Avg ({window})",
                    line=dict(color="orange", width=2, dash="dot")
                ))

            if len(anomaly_times) > 0:
                fig2.add_trace(go.Scatter(
                    x=anomaly_times,
                    y=anomaly_vals,
                    mode="markers",
                    name="Extremes",
                    marker=dict(color="red", size=10, symbol="x")
                ))

            fig2.add_hline(y=mean_val + 2 * std_val, line_dash="dash", line_color="red", opacity=0.5)
            fig2.add_hline(y=mean_val - 2 * std_val, line_dash="dash", line_color="red", opacity=0.5)
            fig2.add_hline(y=mean_val, line_dash="dot", line_color="gray", opacity=0.5)

            display_name = location_pick if LOCATIONS[location_pick][0] is not None else f"({picked_lat}, {picked_lon})"

            fig2.update_layout(
                title=f"{selected_var} at {display_name}",
                xaxis_title="Time",
                yaxis_title=selected_var,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig2)

            st.markdown("---")

            st.subheader("Anomaly Summary")

            if len(anomaly_times) > 0:
                st.warning(f"Found {len(anomaly_times)} extremes (|Z-score| > 2)")
                anomaly_df = pd.DataFrame({
                    "Time": anomaly_times.strftime("%Y-%m-%d"),
                    "Value": anomaly_vals,
                    "Z-Score": z_scores[anomaly_mask]
                })
                anomaly_df["Z-Score"] = anomaly_df["Z-Score"].round(2)
                st.dataframe(anomaly_df)
            else:
                st.success("No extremes detected at this location.")

            st.markdown("---")

            st.subheader("Top 5 Extremes")

            top5_col1, top5_col2 = st.columns(2)

            sorted_high = np.argsort(val_axis)[::-1][:5]
            sorted_low = np.argsort(val_axis)[:5]

            with top5_col1:
                st.markdown("**Highest Values**")
                high_df = pd.DataFrame({
                    "Rank": range(1, 6),
                    "Time": [time_axis[i].strftime("%Y-%m-%d") for i in sorted_high],
                    "Value": [round(val_axis[i], 2) for i in sorted_high]
                })
                st.dataframe(high_df, hide_index=True)

            with top5_col2:
                st.markdown("**Lowest Values**")
                low_df = pd.DataFrame({
                    "Rank": range(1, 6),
                    "Time": [time_axis[i].strftime("%Y-%m-%d") for i in sorted_low],
                    "Value": [round(val_axis[i], 2) for i in sorted_low]
                })
                st.dataframe(low_df, hide_index=True)

            st.markdown("---")

            st.subheader("Distribution View")
            st.caption("Histogram showing how values are spread at this location")

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=val_axis,
                nbinsx=30,
                marker_color="#636EFA",
                opacity=0.8,
                name="Distribution"
            ))

            fig_hist.add_vline(x=mean_val, line_dash="dot", line_color="gray", annotation_text="Mean", annotation_position="top")
            fig_hist.add_vline(x=mean_val + 2 * std_val, line_dash="dash", line_color="red", annotation_text="+2 Std", annotation_position="top")
            fig_hist.add_vline(x=mean_val - 2 * std_val, line_dash="dash", line_color="red", annotation_text="-2 Std", annotation_position="top")

            fig_hist.update_layout(
                xaxis_title=selected_var,
                yaxis_title="Count",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20)
            )

            st.plotly_chart(fig_hist)

            st.markdown("---")

            st.subheader("Auto Insights")

            insights = []

            overall_min = np.nanmin(val_axis)
            overall_max = np.nanmax(val_axis)
            overall_range = overall_max - overall_min
            insights.append(f"Range: {selected_var} varies from {overall_min:.2f} to {overall_max:.2f} (spread of {overall_range:.2f})")

            max_idx = np.nanargmax(val_axis)
            min_idx = np.nanargmin(val_axis)
            insights.append(f"Peak: Highest value of {overall_max:.2f} recorded on {time_axis[max_idx].strftime('%Y-%m-%d')}")
            insights.append(f"Low: Lowest value of {overall_min:.2f} recorded on {time_axis[min_idx].strftime('%Y-%m-%d')}")

            if len(val_axis) > 1:
                first_half = val_axis[:len(val_axis) // 2]
                second_half = val_axis[len(val_axis) // 2:]
                first_mean = np.nanmean(first_half)
                second_mean = np.nanmean(second_half)
                trend_diff = second_mean - first_mean
                if trend_diff > 0:
                    insights.append(f"Trend: {selected_var} increased by {abs(trend_diff):.2f} (comparing first half vs second half of the time range)")
                else:
                    insights.append(f"Trend: {selected_var} decreased by {abs(trend_diff):.2f} (comparing first half vs second half of the time range)")

            if len(anomaly_times) > 0:
                insights.append(f"Extremes: {len(anomaly_times)} statistically unusual events detected (beyond 2 standard deviations)")
            else:
                insights.append("Stability: No significant extremes detected -- data is relatively stable")

            for insight in insights:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

        else:
            st.warning("No time dimension -- cannot plot time-series.")

        st.markdown("---")

        st.subheader("Comparison Mode")
        st.caption("Compare two different time slices side by side")

        if time_dim is not None and time_dim in ds_filtered.dims:
            comp_col1, comp_col2 = st.columns(2)

            time_steps_list = time_steps.tolist()

            with comp_col1:
                st.markdown("**Period A**")
                time_a = st.select_slider("Select Time A", options=time_steps_list, value=time_steps_list[0], key="time_a")

            with comp_col2:
                st.markdown("**Period B**")
                time_b = st.select_slider("Select Time B", options=time_steps_list, value=time_steps_list[-1], key="time_b")

            slice_a = ds_filtered.sel({time_dim: time_a})
            slice_b = ds_filtered.sel({time_dim: time_b})

            z_a = slice_a.values
            z_b = slice_b.values

            if z_a.ndim != 2:
                for rd in [d for d in slice_a.dims if d != lat_name and d != lon_name]:
                    slice_a = slice_a.isel({rd: 0})
                z_a = slice_a.values

            if z_b.ndim != 2:
                for rd in [d for d in slice_b.dims if d != lat_name and d != lon_name]:
                    slice_b = slice_b.isel({rd: 0})
                z_b = slice_b.values

            global_min = min(np.nanmin(z_a), np.nanmin(z_b))
            global_max = max(np.nanmax(z_a), np.nanmax(z_b))

            map_col1, map_col2 = st.columns(2)

            with map_col1:
                fig_a = px.imshow(
                    z_a,
                    x=lon_vals,
                    y=lat_vals,
                    color_continuous_scale="RdBu_r",
                    zmin=global_min,
                    zmax=global_max,
                    labels={"x": "Longitude", "y": "Latitude", "color": selected_var},
                    aspect="auto",
                    origin="lower"
                )
                fig_a.update_layout(
                    title=f"Period A: {pd.Timestamp(time_a).strftime('%Y-%m-%d')}",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_a)

            with map_col2:
                fig_b = px.imshow(
                    z_b,
                    x=lon_vals,
                    y=lat_vals,
                    color_continuous_scale="RdBu_r",
                    zmin=global_min,
                    zmax=global_max,
                    labels={"x": "Longitude", "y": "Latitude", "color": selected_var},
                    aspect="auto",
                    origin="lower"
                )
                fig_b.update_layout(
                    title=f"Period B: {pd.Timestamp(time_b).strftime('%Y-%m-%d')}",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_b)

            st.markdown("**Difference (B - A)**")

            diff_vals = z_b - z_a

            fig_diff = px.imshow(
                diff_vals,
                x=lon_vals,
                y=lat_vals,
                color_continuous_scale="RdBu_r",
                labels={"x": "Longitude", "y": "Latitude", "color": "Difference"},
                aspect="auto",
                origin="lower"
            )
            fig_diff.update_layout(
                height=400,
                margin=dict(l=60, r=60, t=30, b=20)
            )

            pad5, diff_main, pad6 = st.columns([0.3, 5, 0.3])
            with diff_main:
                st.plotly_chart(fig_diff)

            diff_mean = np.nanmean(diff_vals)
            if diff_mean > 0:
                st.markdown(f'<div class="insight-card">Average change from A to B: +{diff_mean:.2f} (warming/increase)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-card">Average change from A to B: {diff_mean:.2f} (cooling/decrease)</div>', unsafe_allow_html=True)

        else:
            st.warning("No time dimension -- cannot compare periods.")

        st.markdown("---")

        st.subheader("Disaster Explorer")

        disaster_mode = st.radio("Select Mode", ["Historical Events", "Live Events (NASA EONET)"], horizontal=True)

        if disaster_mode == "Historical Events":
            active_list = []
            if time_dim is not None and time_dim in ds_filtered.dims:
                for disaster in DISASTERS:
                    d_time = pd.Timestamp(disaster["time"])
                    if time_min <= d_time <= time_max:
                        active_list.append(disaster)

            if len(active_list) == 0:
                active_list = DISASTERS

            card_class = "disaster-card"
            title_class = "disaster-title"

        else:
            live_events = fetch_live_events()
            if len(live_events) > 0:
                active_list = live_events
            else:
                st.warning("Could not fetch live events. Showing historical instead.")
                active_list = DISASTERS

            card_class = "live-card"
            title_class = "live-title"

        if len(active_list) > 0:
            if "disaster_index" not in st.session_state:
                st.session_state["disaster_index"] = 0

            if st.session_state["disaster_index"] >= len(active_list):
                st.session_state["disaster_index"] = 0

            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

            with nav_col1:
                if st.button("Previous", key="prev_disaster"):
                    if st.session_state["disaster_index"] > 0:
                        st.session_state["disaster_index"] -= 1

            with nav_col3:
                if st.button("Next", key="next_disaster"):
                    if st.session_state["disaster_index"] < len(active_list) - 1:
                        st.session_state["disaster_index"] += 1

            current = active_list[st.session_state["disaster_index"]]

            with nav_col2:
                st.markdown(f"**Event {st.session_state['disaster_index'] + 1} of {len(active_list)}**")

            st.markdown(f"""
            <div class="{card_class}">
                <div class="{title_class}">{current["name"]}</div>
                <strong>Category:</strong> {current["category"]}<br>
                <strong>Location:</strong> ({current["lat"]}, {current["lon"]})<br>
                <strong>Date:</strong> {current["time"]}<br><br>
                <strong>Impact:</strong> {current["impact"]}
            </div>
            """, unsafe_allow_html=True)

            if lat_name and lon_name:
                nearest_lat = min(lat_list, key=lambda x: abs(x - current["lat"]))
                nearest_lon = min(lon_list, key=lambda x: abs(x - current["lon"]))

                if time_dim is not None and time_dim in ds_filtered.dims:
                    d_time = pd.Timestamp(current["time"])
                    nearest_time = min(time_steps, key=lambda x: abs(x - d_time))

                    disaster_slice = ds_filtered.sel({time_dim: nearest_time})
                    disaster_z = disaster_slice.values

                    if disaster_z.ndim != 2:
                        for rd in [dd for dd in disaster_slice.dims if dd != lat_name and dd != lon_name]:
                            disaster_slice = disaster_slice.isel({rd: 0})
                        disaster_z = disaster_slice.values

                    fig_disaster_map = px.imshow(
                        disaster_z,
                        x=lon_vals,
                        y=lat_vals,
                        color_continuous_scale="RdBu_r",
                        labels={"x": "Longitude", "y": "Latitude", "color": selected_var},
                        aspect="auto",
                        origin="lower"
                    )

                    fig_disaster_map.add_trace(go.Scatter(
                        x=[nearest_lon],
                        y=[nearest_lat],
                        mode="markers+text",
                        marker=dict(color="black", size=15, symbol="circle-open", line=dict(width=3)),
                        text=[current["name"]],
                        textposition="top center",
                        textfont=dict(color="black", size=12),
                        showlegend=False
                    ))

                    fig_disaster_map.update_layout(
                        title=f"{current['name']} -- {pd.Timestamp(nearest_time).strftime('%Y-%m-%d')}",
                        height=450,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )

                    st.plotly_chart(fig_disaster_map)

                    disaster_point = ds_filtered.sel(
                        {lat_name: nearest_lat, lon_name: nearest_lon},
                        method="nearest"
                    )
                    extra_dims = [dd for dd in disaster_point.dims if dd != time_dim]
                    for ed in extra_dims:
                        disaster_point = disaster_point.isel({ed: 0})

                    d_time_axis = pd.to_datetime(disaster_point.coords[time_dim].values)
                    d_val_axis = disaster_point.values

                    fig_disaster_line = go.Figure()
                    fig_disaster_line.add_trace(go.Scatter(
                        x=d_time_axis,
                        y=d_val_axis,
                        mode="lines",
                        name=selected_var,
                        line=dict(color="#636EFA")
                    ))

                    fig_disaster_line.add_vline(
                        x=nearest_time.timestamp() * 1000,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=current["name"],
                        annotation_position="top"
                    )

                    fig_disaster_line.update_layout(
                        title=f"Time-series at event location ({nearest_lat}, {nearest_lon})",
                        xaxis_title="Time",
                        yaxis_title=selected_var,
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )

                    st.plotly_chart(fig_disaster_line)

        st.markdown("---")

        st.subheader("3D Globe View")

        globe_step = max(1, len(lat_vals) // 30)
        lat_globe = lat_vals[::globe_step]
        lon_globe = lon_vals[::globe_step]
        z_globe = z_vals[::globe_step, ::globe_step]

        globe_lats = []
        globe_lons = []
        globe_vals = []

        for i in range(len(lat_globe)):
            for j in range(len(lon_globe)):
                val = z_globe[i, j]
                if np.isnan(val):
                    continue
                globe_lats.append(float(lat_globe[i]))
                globe_lons.append(float(lon_globe[j]))
                globe_vals.append(float(val))

        fig_globe = go.Figure(data=go.Scattergeo(
            lat=globe_lats,
            lon=globe_lons,
            marker=dict(
                size=4,
                color=globe_vals,
                colorscale="RdBu_r",
                colorbar=dict(title=selected_var),
                opacity=0.8
            ),
            mode="markers"
        ))

        fig_globe.update_geos(
            projection_type="orthographic",
            showland=True,
            landcolor="#e0e0e0",
            showocean=True,
            oceancolor="#d4e8f0",
            showcoastlines=True,
            coastlinecolor="#999"
        )

        fig_globe.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig_globe)

        st.markdown("---")

        st.subheader("Export Data")

        if time_dim is not None and time_dim in ds_filtered.dims:
            export_df = pd.DataFrame({
                "Time": time_axis.strftime("%Y-%m-%d"),
                selected_var: val_axis
            })

            st.dataframe(export_df)

            csv_data = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{selected_var}_lat{picked_lat}_lon{picked_lon}.csv",
                mime="text/csv"
            )

    else:
        st.warning("Could not find latitude/longitude dimensions.")
