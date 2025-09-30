import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import csv

def convert_and_export_weather_data():
    """
    Reads weather data from an XML file, converts it into classification and 
    regression datasets, and exports them into two separate CSV files.
    """
    # --- 1. Read and parse the source XML data ---
    try:
        script_dir = Path(__file__).parent.resolve()
        xml_file = script_dir / 'O-A0038-003.xml'
        tree = ET.parse(xml_file)
        root = tree.getroot()
        namespace = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}
        content_str = root.find('.//cwa:Content', namespace).text
        lines = content_str.strip().split('\n')
        all_floats = []
        for line in lines:
            if not line.strip():
                continue
            floats_in_line = [float(val) for val in line.split(',') if val.strip()]
            all_floats.extend(floats_in_line)
        temp_grid = np.array(all_floats).reshape(120, 67)
    except FileNotFoundError:
        print(f"Error: File not found at '{xml_file}'. Please ensure it is in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred while reading or parsing the XML file: {e}")
        return

    # --- 2. Create the latitude and longitude coordinate grid ---
    start_lon = 120.00
    start_lat = 21.88
    lon_resolution = 0.03
    lat_resolution = 0.03
    lon_points = 67
    lat_points = 120
    longitudes = start_lon + np.arange(lon_points) * lon_resolution
    latitudes = start_lat + np.arange(lat_points) * lat_resolution

    # --- 3. Generate the classification and regression datasets ---
    classification_data = []
    regression_data = []
    for i in range(lat_points):
        for j in range(lon_points):
            lon = longitudes[j]
            lat = latitudes[i]
            temp_value = temp_grid[i, j]
            label = 1 if temp_value != -999.0 else 0
            classification_data.append({'longitude': lon, 'latitude': lat, 'label': label})
            if temp_value != -999.0:
                regression_data.append({'longitude': lon, 'latitude': lat, 'value': temp_value})
    
    print("Data conversion complete.")
    print(f"Total entries in classification dataset: {len(classification_data)}.")
    print(f"Total entries in regression dataset: {len(regression_data)}.")

    # --- 4. Write the data into two separate CSV files ---

    # (a) Write the classification dataset
    classification_csv_file = script_dir / "classification_data.csv"
    try:
        with open(classification_csv_file, 'w', newline='', encoding='utf-8') as f:
            # Define the fieldnames (CSV header)
            fieldnames = ['longitude', 'latitude', 'label']
            # Create a DictWriter object to map dictionaries to CSV rows
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()  # Write the header row
            writer.writerows(classification_data) # Write all data rows
            
        print(f"\nSuccessfully wrote classification data to: '{classification_csv_file}'")
    except Exception as e:
        print(f"An error occurred while writing the classification CSV: {e}")

    # (b) Write the regression dataset
    regression_csv_file = script_dir / "regression_data.csv"
    try:
        with open(regression_csv_file, 'w', newline='', encoding='utf-8') as f:
            # Define the fieldnames
            fieldnames = ['longitude', 'latitude', 'value']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader() # Write the header row
            writer.writerows(regression_data) # Write all data rows
            
        print(f"Successfully wrote regression data to: '{regression_csv_file}'")
    except Exception as e:
        print(f"An error occurred while writing the regression CSV: {e}")


if __name__ == '__main__':
    convert_and_export_weather_data()