import pandas as pd
import argparse

def convert_to_datatime(input_csv):
    
    # Read the data from the CSV file
    df = pd.read_csv(input_csv)

    # Convert the timestamp column to datetime format
    df['timestamputc'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Format the timestamp column to the desired format
    df['timestamputc'] = df['timestamputc'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save the updated DataFrame to a new CSV file
    df.to_csv('data_with_formatted_timestamps.csv', index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv_input_file', required=True, type=str, help='Path to the reference folder')
  args = parser.parse_args()

  convert_to_datatime(args.csv_input_file)
