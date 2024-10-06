# Seismic Waves Detection App 🔭

Developed for **NASA Space Apps Challenge** by Team **Astro Boys**

Welcome to the Seismic Waves Detection App! This tool is designed to simplify the process of detecting and analyzing seismic waves, especially the precise P-wave arrival time, which is key in fields like seismology and earthquake research. Built for the NASA Space Apps Challenge, this app makes seismic data analysis accessible and interactive, right from your browser!

![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/00.png)

## How It Works?

Once you upload the file, the app will display file details right away so you can confirm that everything’s good to go.
![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/1.png)
From there, the app guides you through these steps:

- At each stage, the app offers visualizations so you can see exactly what’s happening to your data

### Filtering the Signal

You’ll start by applying a high-pass filter that cleans the raw data by removing unwanted noise.

![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/3.png)

### Envelope Extraction

By applying a Hilbert transform, the app captures the envelope of the waveform, making it easy to spot significant features.

![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/4.png)

### Anamoly Detection

Here, you’ll see peaks highlighted directly on the signal envelope, with the app doing the heavy lifting to detect them for you.
![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/5.png)

### Clustering of the anomalies

Using K-means clustering, the app groups the peaks and displays the densest cluster visually, so you can zoom in on the most relevant wave segments.
![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/6.png)

### Outlier Detection and Removal

The app automatically removes outliers in the densest cluster, ensuring the P-wave detection is as accurate as possible.
![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/7.png)

### P-wave Detection time

Finally, the app marks the P-wave arrival time on the original signal, helping you pinpoint it with confidence.

![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/8.png)

## Getting Started

Upload Your Seismic Data: Upload your MiniSEED file (.mseed), and you’re ready to go.

- **Step-by-Step Processing:** Follow the instructions in the sidebar, each step unlocking only after the previous one is complete.
- **Interactive Visualizations:** Each step includes clear visuals so you can track the progress and results at each stage.

## Built With

- **Languages & Libraries:** Python, Streamlit, NumPy, SciPy, ObsPy, Plotly, scikit-learn
- **Data Format:** MiniSEED (.mseed) files

# Team Astro Boys

![App Screenshot](https://github.com/OmarMDiab/NASA-SPACE-APPS-CHALLENGE-2024/blob/main/img/team.jpg)
