import streamlit as st
import xarray as xr
import tempfile
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(page_title="PyClimaExplorer", page_icon="", layout="wide")

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">PyClimaExplorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Climate Data Dashboard</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a NetCDF file", type=["nc"])

if uploaded_file is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    ds = xr.open_dataset(tmp.name)
    ds.load()
    ds.close()
    os.unlink(tmp.name)

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

        selected_time = control_col2.slider(
            "Select Time Range",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max)
        )

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
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig)

        st.markdown("---")

        st.subheader("Temporal View")
        st.caption("Drag the sliders to pick a location from the map above")

        if time_dim is not None and time_dim in ds_filtered.dims:
            loc_col1, loc_col2 = st.columns(2)

            lat_list = sorted(lat_vals.tolist())
            lon_list = sorted(lon_vals.tolist())

            picked_lat = loc_col1.select_slider("Select Latitude", options=lat_list, value=lat_list[len(lat_list) // 2])
            picked_lon = loc_col2.select_slider("Select Longitude", options=lon_list, value=lon_list[len(lon_list) // 2])

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

            if len(anomaly_times) > 0:
                fig2.add_trace(go.Scatter(
                    x=anomaly_times,
                    y=anomaly_vals,
                    mode="markers",
                    name="Anomaly (|Z| > 2)",
                    marker=dict(color="red", size=10, symbol="x")
                ))

            fig2.add_hline(y=mean_val + 2 * std_val, line_dash="dash", line_color="red", opacity=0.5)
            fig2.add_hline(y=mean_val - 2 * std_val, line_dash="dash", line_color="red", opacity=0.5)
            fig2.add_hline(y=mean_val, line_dash="dot", line_color="gray", opacity=0.5)

            fig2.update_layout(
                title=f"{selected_var} at ({picked_lat}, {picked_lon})",
                xaxis_title="Time",
                yaxis_title=selected_var,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig2)

            st.markdown("---")

            st.subheader("Anomaly Summary")

            if len(anomaly_times) > 0:
                st.warning(f"Found {len(anomaly_times)} anomalies (|Z-score| > 2)")
                anomaly_df = pd.DataFrame({
                    "Time": anomaly_times.strftime("%Y-%m-%d"),
                    "Value": anomaly_vals,
                    "Z-Score": z_scores[anomaly_mask]
                })
                anomaly_df["Z-Score"] = anomaly_df["Z-Score"].round(2)
                st.dataframe(anomaly_df)
            else:
                st.success("No anomalies detected at this location.")

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
                insights.append(f"Anomalies: {len(anomaly_times)} statistically unusual events detected (beyond 2 standard deviations)")
            else:
                insights.append("Stability: No significant anomalies detected -- data is relatively stable")

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
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_diff)

            diff_mean = np.nanmean(diff_vals)
            if diff_mean > 0:
                st.markdown(f'<div class="insight-card">Average change from A to B: +{diff_mean:.2f} (warming/increase)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-card">Average change from A to B: {diff_mean:.2f} (cooling/decrease)</div>', unsafe_allow_html=True)

        else:
            st.warning("No time dimension -- cannot compare periods.")

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
                selected_var: val_axis,
                "Z-Score": z_scores.round(2),
                "Is_Anomaly": anomaly_mask
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

else:
    st.info("Upload a .nc file to get started.")