import streamlit as st
from scipy.signal import correlate, find_peaks,butter, filtfilt
from scipy.signal import hilbert, find_peaks
import numpy as np 
from obspy import read
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import pandas as pd
import graphviz as graphviz

# filter the data
def highpass(data,cutoff , fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    high = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, high, btype='high')  # High-pass Butterworth filter
    filtered_data = filtfilt(b, a, data)  # Apply the filter using filtfilt for zero-phase filtering
    return filtered_data


# function to make all the data +ve only
def make_positive(data):
    return np.abs(data)


def main():
    st.title("Seismic Waves Detection App 🔭")
    with st.sidebar:
        st.header("Developed by Astro Boys 🚀")
        st.write("- Abdelrahman Omran")
        st.write("- Omar Diab")
        st.write("- Ahmed Abdelaty")
        st.write("- Adham Kandil")
        st.write("- Abdelrahman Diaa")
        st.write("---------------------------------")
        st.subheader("Approach Steps")
        st.write("""
                1. **Filter**: High-pass filter to remove noise.
                2. **Normalize**: Convert signal to positive values.
                3. **Envelope**: Calculate using Hilbert transform.
                4. **Peak Detection**: Identify peaks by prominence.
                5. **Clustering**: Use K-means to find densest clusters.
                6. **Outlier Handling**: Remove outliers using IQR.
                7. **P-wave Detection**: Determine first peak in densest cluster.

                - **Output**: Visualizations of signals, peaks, clusters, and detected P-wave arrival time.
                """)
    st.write("---------------------------------")
    file_path = st.file_uploader("Upload the event file", type=["mseed"])
    if file_path is not None:
        # write progress bar
        st.subheader("File Details")

        # read the file
        wv = read(file_path)
        tr = wv[0]
        st.write(tr.stats)

        # Get important values
        delta = tr.stats.delta
        npts = tr.stats.npts
        times_rel = np.asarray([i * delta for i in range(npts)])

        st.write("-----------------------------")
        st.write("### 1) Original Signal")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=times_rel, y=tr.data, mode='lines', name='Original Signal', line=dict(color='firebrick')))
        fig1.update_layout(title="Original Seismic Wave", xaxis_title="Time (s)", plot_bgcolor="white",yaxis_title="Amplitude")
        st.plotly_chart(fig1)

        st.write("-----------------------------")
        st.write("### 2) Filter Out the noise ")   
        # Apply bandpass filter
        bandpass_Wave = highpass(tr.data,0.5 , fs=tr.stats.sampling_rate)
    
        # Plot original and filtered data using Plotly

        fig = go.Figure()

        # Original signal
        fig.add_trace(go.Scatter(x=times_rel, y=tr.data, mode='lines', name='Original Signal', line=dict(color='firebrick', width=2)))

        # Filtered signal
        fig.add_trace(go.Scatter(x=times_rel, y=bandpass_Wave, mode='lines', name='Highpass Filtered Signal', line=dict(color='green', width=2)))

        # Set plot layout for better presentation
        fig.update_layout(
            title="Original vs Highpass Filtered Signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            legend=dict(x=0.01, y=0.99),
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified",
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # Make the data positive
        positive_wave = make_positive(bandpass_Wave)



        st.write("-----------------------------")
        st.write("### 3) Envelope of the Signal")
        # Calculate envelope using Hilbert transform
        envelope = np.abs(hilbert(positive_wave))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=times_rel, y=envelope, mode='lines', name='Envelope', line=dict(color='purple', width=2)))
        fig2.update_layout(
            title="Envelope of the Signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            legend=dict(x=0.01, y=0.99),
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

        peaks =[]
        prom = (1e-9)+(1e-10) # initial prominence value
        st.write("-----------------------------")
        st.write("### 4) Finding Anomalies in the data 🔍")

        while(len(peaks)<10): # if no peaks found, change the prominence value
                # Find peaks using the percentile threshold
            peaks, _ = find_peaks(envelope, height=None, prominence=(prom))
            if (len(peaks)<10):

                prom -=1e-11
                peaks, _ = find_peaks(envelope, height=None, prominence=(prom))

        # Create Plotly figure
        fig3 = go.Figure()

        # Plot the envelope
        fig3.add_trace(go.Scatter(x=times_rel, y=envelope, mode='lines', name='Envelope', line=dict(color='blue', width=2)))

            # Plot peaks
        fig3.add_trace(go.Scatter(x=times_rel[peaks], y=envelope[peaks], mode='markers', name='Peaks', marker=dict(color='red', size=8)))

        # Update layout for better presentation
        fig3.update_layout(
            title="Envelope with Peaks",
            xaxis_title="Relative Time (seconds)",
            yaxis_title="Amplitude",
            legend=dict(x=0.01, y=0.99),
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified",
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig3, use_container_width=True)

        peak_times = times_rel[peaks]
        
        # Get first peak
        First_Peak = times_rel[peaks[0]]
        
        ptime = First_Peak - 100 # initially
        
        # Clustering peaks (optional)
        peak_times_reshaped = peak_times.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=15, random_state=0)
        labels = kmeans.fit_predict(peak_times_reshaped)
        
        # Find the densest cluster and the first peak in that cluster
        cluster_counts = np.bincount(labels)
        densest_cluster_label = np.argmax(cluster_counts)
        densest_cluster_peak_times = peak_times[labels == densest_cluster_label]
# --------------------------------------------------------------------------------------------------
        st.write("-----------------------------")
        st.write("### 5) Clustering the Peaks of envelope 🤖")
        # Create a Plotly figure for the envelope and clusters
        fig_clusters_on_envelope = go.Figure()

        # Plot the envelope
        fig_clusters_on_envelope.add_trace(go.Scatter(
            x=times_rel,
            y=envelope,
            mode='lines',
            name='Envelope',
            line=dict(color='blue', width=2)
        ))

        # Plot the peaks, color-coded by their cluster labels
        for cluster_id in np.unique(labels):
            cluster_peaks = peak_times[labels == cluster_id]
            cluster_peak_envelope_values = envelope[peaks][labels == cluster_id]  # Get the envelope values for the peaks in this cluster
            fig_clusters_on_envelope.add_trace(go.Scatter(
                x=cluster_peaks,
                y=cluster_peak_envelope_values,
                mode='markers',
                marker=dict(size=10, opacity=0.8),
                name=f'Cluster {cluster_id}'
            ))

        # Highlight the densest cluster's first peak
        fig_clusters_on_envelope.add_trace(go.Scatter(
            x=[densest_cluster_peak_times[0]],  # X value for the first peak in the densest cluster
            y=[envelope[peaks][labels == densest_cluster_label][0]],  # Corresponding Y value from the envelope
            mode='markers',
            marker=dict(color='gold', size=12, symbol='star-triangle-up'),
            name='First Peak in Densest Cluster'
        ))


        # Update layout for better presentation
        fig_clusters_on_envelope.update_layout(
            title="K-means Clustering of Peaks on Envelope",
            xaxis_title="Relative Time (seconds)",
            yaxis_title="Amplitude",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified",
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig_clusters_on_envelope, use_container_width=True)




                # >>>>>>> Handling_Outliers

        st.write("-----------------------------")
        st.write("### 6) Handling choosen Cluster's Outliers using Interquartile Range (IQR) 🦅")
        fig4 = go.Figure()
        fig4.add_trace(go.Box(y=densest_cluster_peak_times, name='Peaks', marker=dict(color='blue')))
        fig4.update_layout(
            title="Box-plot of the most dense cluster peaks",
            yaxis_title="Time (s)",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
        )

        st.plotly_chart(fig4, use_container_width=True)
        Q1 = np.percentile(densest_cluster_peak_times, 25)
        Q3 = np.percentile(densest_cluster_peak_times, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = np.where((densest_cluster_peak_times < lower_bound) | (densest_cluster_peak_times > upper_bound))[0]
        densest_cluster_peak_times = densest_cluster_peak_times[(densest_cluster_peak_times >= lower_bound) & (densest_cluster_peak_times <= upper_bound)]
        # plot the box plot of the densest cluster
        if len(outliers)==0:
            st.success("No Outliers Found")
            
        else:
            st.success("Outliers removed!")
            st.table({
            "Description": ["First Peak", "First Peak in Densest Cluster", "Outliers", "First Peak in Densest Cluster after removing outliers"],
            "Value": [
            f"{First_Peak:.2f} s",
            f"{densest_cluster_peak_times[0]:.2f} s",
            f"{outliers}",
            f"{densest_cluster_peak_times[0]:.2f} s"
            ]
        })

        # Create a table to display the results
        first_peak_in_Densest_cluster = densest_cluster_peak_times[0] - 100
        # plot the original wave with the time of first_peak_in the densest cluster
        st.write("-----------------------------")
        st.write("### 7) P-wave Detection 🚀")
        st.write("- **The first point** of the densest cluster (after removing outliers) is considered as the P-wave arrival time")
        # Create a Plotly figure for the original wave
        fig_original_wave = go.Figure()

        # Plot the original signal
        fig_original_wave.add_trace(go.Scatter(
            x=times_rel,
            y=tr.data,  # Original wave
            mode='lines',
            name='Original Signal',
            line=dict(color='green', width=1)
        ))

        # Add the vertical line for ptime
        fig_original_wave.add_trace(go.Scatter(
            x=[first_peak_in_Densest_cluster, first_peak_in_Densest_cluster],  # X value for the vertical line
            y=[min(tr.data), max(tr.data)],  # Y values from min to max of the original signal
            mode='lines',
            line=dict(color='red', dash='solid', width=2),
            name='Start of P-wave'
        ))

        # Update layout for better presentation
        fig_original_wave.update_layout(
            title="Original Wave with Start of P-wave",
            xaxis_title="Relative Time (seconds)",
            yaxis_title="Amplitude",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified",
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig_original_wave, use_container_width=True)
        Arrival_time = int(first_peak_in_Densest_cluster)
        absolute_time = (tr.stats.starttime + Arrival_time).strftime("%Y-%m-%dT%H:%M:%S.%f")
        st.success(f"P-wave detected at relative time **[{Arrival_time}]** seconds and at absolute time **[{absolute_time}]**")

        
if __name__ == "__main__":
    main()  # calling the main function