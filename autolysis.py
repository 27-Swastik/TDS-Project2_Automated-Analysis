
# uv:
# dependencies:
#   - pandas
#   - seaborn
#   - matplotlib
#   - scikit-learn
#   - numpy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import random
import os, sys
import math
import requests, json, ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D





def make_api_call(prompt):
    #Fetch API key from environment

    try:
        api_key = os.environ['AIPROXY_TOKEN']
    except KeyError:
        print("AIPROXY_TOKEN environment variable not found.")
        api_key = None

    if not api_key:
        print("API key is missing. Cannot make API call.")
        return None

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    data = {"model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": prompt}
            ]}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error making API call: {e}")
        if e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None


def load_csv(file_path, encodings=None):
    """
    Tries to load a CSV file into a Pandas DataFrame using different encodings.

    Args:
        file_path (str): Path to the CSV file.
        encodings (list, optional): List of encodings to try. Defaults to common encodings.

    Returns:
        pd.DataFrame: Loaded DataFrame if successful.
        str: Error message if all attempts fail.
    """
    if encodings is None:
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except Exception as e:
            None

    return "All attempts to load the file failed. Please check the file format and content."



def analyze_data_for_relationships(df, sample_size=10):
    """Analyzes a dataset by sending a sample to the LLM for variable relationships."""
    
    # Sample the first few rows of the dataframe (or randomly sample if desired)
    sample_data = df.sample(sample_size)
    
    # Convert the sample data to a string for easy consumption by the LLM
    sample_str = sample_data.to_string(index=False)
    
    # Create a prompt asking the LLM to analyze the dataset
    prompt = (
        f"I have a dataset with the following data:\n{sample_str}\n\n"
        "Please analyze the dataset and suggest the most optimal variables (max 8, min 4) whose relationships "
        "could be studied using correlation heat maps and distribution plots(if numeric),"
        "clustering analysis (if geographic data e.g. latitudes or longitudes, or catagorical data),"
         " or time series plots.(for date type data). Provide just a list of those variables with no other surrounding text."
         "Keep the format of your response strictly like: ['column1', 'column2'] ONLY."
         "Try to have at least 2 catagorical data types in your response."
    )
    
    # Make the API call
    return make_api_call(prompt)




def preprocess_dataframe(df):
    """
    Prepares a summary of the DataFrame for LLM analysis.

    Args:
        df (pd.DataFrame): The full dataset.

    Returns:
        dict: A summary with column metadata and a sample of the data.
    """

    variables = ast.literal_eval(analyze_data_for_relationships(df)) #ask the llm for apropriate variables

    summary = {}
    for column in variables:
        col_data = df[column]
        col_summary = {
            "data_type": str(col_data.dtypes),
            "unique_values": len(col_data.unique()),
            "sample_values": col_data.sample(min(len(col_data), 5)).tolist(),  # Random 5 samples
            "null_count": col_data.isnull().sum(),
            "mean": col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else None,
            "std": col_data.std() if pd.api.types.is_numeric_dtype(col_data) else None,
            "mode": col_data.mode().iloc[0] if not col_data.mode().empty else None
        }
        summary[column] = col_summary

    return summary




def classify_columns(summary):
    """
    Classifies DataFrame columns based on their metadata.

    Args:
        summary (dict): Summary metadata of the DataFrame.

    Returns:
        dict: Classified columns under analysis types.
    """
    classifications = {
        "Categorical": [],
        "Numerical": [],
        "Time Series": [],
        "Geographic": [],
        "Descriptive": []
    }

    for column, metadata in summary.items():
        # Attempt to identify numerical data even if stored as strings or marked as general type
        if metadata["data_type"] in ["int64", "float64"] or \
           (metadata["data_type"] == "object" and all(str(value).replace('.', '', 1).isdigit() for value in metadata["sample_values"] if pd.notnull(value))):
            try:
                # Attempt to coerce to numeric to confirm numerical data
                coerced_values = pd.to_numeric(metadata["sample_values"], errors='coerce')
                if pd.notnull(coerced_values).all():
                    if metadata["unique_values"] < 0.5 * len(metadata["sample_values"]):
                        classifications["Categorical"].append(column)
                    else:
                        classifications["Numerical"].append(column)
                else:
                    classifications["Categorical"].append(column)
            except ValueError:
                classifications["Categorical"].append(column)
        elif metadata["data_type"] == "object":
            if any(keyword in column.lower() for keyword in ["lat", "long", "city", "district", "state", "country"]):
                classifications["Geographic"].append(column)
            elif any(keyword in column.lower() for keyword in ["date"]):
                classifications["Time Series"].append(column)
            else:
                classifications["Descriptive"].append(column)
        elif "datetime" or "date" in metadata["data_type"].lower():
            classifications["Time Series"].append(column)

    return classifications


#--------------------------------------------------------------------------------------------------
def analyze_numerical(df, columns, output_folder):
    """Generates a correlation heatmap for numerical columns with improved spacing and design."""
    if len(columns) < 2:
        print("Not enough numerical columns for correlation analysis.")
        return

    # Compute the correlation matrix
    corr_matrix = df[columns].corr()

    # Set up the matplotlib figure with improved size
    plt.figure(figsize=(10, 8))

    # Create the heatmap with better spacing and design
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, 
                cbar_kws={'shrink': 0.8}, annot_kws={"size": 10}, square=True)

    # Rotate the tick labels to improve readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Set title with some padding
    plt.title("Correlation Heatmap", fontsize=16, pad=20)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save plot to the specified output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(f'{output_folder}/correlation_heatmap.png')
    plt.close()



def analyze_categorical_or_geographic(df, columns, output_folder):
    """
    Performs clustering analysis and plots the results,
    reducing dimensions with PCA if necessary.
    Handles missing values appropriately.
    """
    if not columns:
        print("No columns provided for clustering.")
        return

    # Check if columns are in the dataframe
    columns = [col for col in columns if col in df.columns]
    if not columns:
        print("None of the specified columns are in the dataframe.")
        return

    # Encode categorical columns
    le = LabelEncoder()
    encoded_data = df[columns].apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')  # Use 'mean' for numerical data
    encoded_data = pd.DataFrame(imputer.fit_transform(encoded_data), columns=columns)

    # Handle dimensionality
    n_columns = len(columns)

    if n_columns == 1:
        print(f"Only one column '{columns[0]}' available. Clustering is not meaningful for 1D data.")
        return
    elif n_columns == 2:
        # Two columns, no PCA needed
        data = encoded_data.values
        print("Two columns available. Clustering in 2D.")
    else:
        # More than 3 columns, apply PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(encoded_data)
        pca = PCA(n_components=3)
        data = pca.fit_transform(scaled_data)
        print(f"Applied PCA to reduce {n_columns} columns to 3 dimensions.")

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data)

    # Visualization
    if n_columns == 1:
        # 1D Plot
        plt.figure(figsize=(8, 4))
        plt.scatter(data, [0] * len(data), c=clusters, cmap='viridis', s=100)
        plt.xlabel(columns[0])
        plt.title("1D Clustering Visualization")
    elif n_columns == 2:
        # 2D Plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=100)
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.title("2D Clustering Visualization")
        plt.colorbar(scatter)
    else:
        # 3D Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=clusters, cmap='viridis', s=100)
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        ax.set_title("3D Clustering Visualization")
        plt.colorbar(scatter)

    plt.tight_layout()

    # Save plot to the specified output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(f'{output_folder}/clustering_visualization.png')

    plt.close()


def analyze_time_series(df, time_series_columns, numerical_columns, output_folder):
    """Plots time series graphs for all numerical variables against the time column."""

    if not time_series_columns:
        print("No time series columns available for analysis.")
        return

    time_column = time_series_columns[0]

    # Ensure time column is numeric or convert it to datetime
    if not pd.api.types.is_numeric_dtype(df[time_column]):
        try:
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce', dayfirst=True)
            if df[time_column].isnull().any():
                print(f"Warning: Some date entries in '{time_column}' could not be parsed and were set to NaT (Not a Time).")
        except Exception as e:
            print(f"Error converting '{time_column}' to datetime: {e}")
            return

    # Ensure there are numerical columns to plot
    numerical_columns = [col for col in numerical_columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numerical_columns:
        print("No valid numerical columns available for plotting.")
        return

    # Determine the number of rows and columns for subplots
    num_plots = len(numerical_columns)
    num_rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # Flatten axes array for easy indexing if there's more than one row
    axes = axes.flatten()

    # Plot each numerical column against the time column
    for i, column in enumerate(numerical_columns):
        axes[i].plot(df[time_column], df[column], label=column, color=sns.color_palette("husl", num_plots)[i], linestyle='-', marker='o')
        axes[i].set_xlabel(str(time_column), fontsize=12)
        axes[i].set_ylabel(str(column), fontsize=12)
        axes[i].set_title(f"Time Series: {column} vs {time_column}", fontsize=14)
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Hide unused subplots if the number of numerical columns is less than the number of axes
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save plot to the specified output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(f'{output_folder}/time_series_visualization.png')

    plt.close()


def plot_numerical_distributions(df, numerical_columns, output_folder):
    """
    Plots the distributions of numerical variables in the dataframe with a maximum of 3 plots per row.
    If the dataframe has more than 1000 rows, a random sample of 1000 rows is used to speed up the computation.

    Parameters:
    - df: pandas.DataFrame
    - numerical_columns: list of strings (column names of numerical variables)
    """
    if not numerical_columns:
        print("No numerical columns provided for plotting.")
        return

    # Check if all columns exist in the DataFrame
    numerical_columns = [col for col in numerical_columns if col in df.columns]

    if not numerical_columns:
        print("None of the provided columns exist in the DataFrame.")
        return

    # If the dataframe has more than 1000 rows, sample it
    if len(df) > 1000:
        print(f"Data has large number of rows, randomly sampling 2000 rows for plotting to save computation time and resources.")
        df = df.sample(n=1000, random_state=42)  # Set a random_state for reproducibility

    # Calculate the number of rows needed (3 plots per row)
    num_plots = len(numerical_columns)
    num_rows = math.ceil(num_plots / 3)

    # Create subplots with a max of 3 plots per row
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # Flatten axes array to make it easier to iterate
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        sns.histplot(df[column], kde=True, color=sns.color_palette("husl", num_plots)[i], ax=axes[i])
        axes[i].set_title(f"Distribution of {column}", fontsize=14)
        axes[i].set_xlabel(column, fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)
        axes[i].grid(visible=True, linestyle='--', alpha=0.7)

    # Hide any unused axes
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save plot to the specified output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(f'{output_folder}/numerical_distributions.png')

    plt.close()

#----------------------------------------------------------------------------------------



# Function to sample data and generate an analysis prompt
def generate_llm_analysis(df, num_rows=100):
    """
    Sends a sample of the dataset to the LLM and asks for analysis based on historical analogies,
    hypothesis generation, and what-if scenarios.

    Parameters:
    - df: pandas.DataFrame, the dataset to analyze
    - num_rows: int, the number of rows to sample for analysis

    Returns:
    - response: str, the LLM response
    """
    # Ensure the sample size does not exceed the dataset size
    sample_size = min(num_rows, len(df))

    # Randomly sample rows from the dataset
    sampled_data = df.sample(n=sample_size, random_state=random.randint(1, 1000))

    # Convert sampled data to a dictionary for a compact representation
    sampled_data_dict = sampled_data.to_dict(orient="records")

    # Construct the LLM prompt

    
    prompt = (
        "You are an advanced data analyst. Here is a sample of the dataset: \n" +
        f"{sampled_data_dict}\n" +
        "Analyze the dataset under the following guidelines, providing evidence from the data:\n"
        "1. Identify meaningful patterns or trends using historical analogies or futuristic projections.\n"
        "2. Formulate hypotheses about the data (e.g., 'Do more experienced users tend to make fewer errors?').\n"
        "3. Simulate 'what-if' scenarios (e.g., 'What happens if sales grow by 20% next quarter?').\n"
        '''4. Generate multidimensional analyses: if the data includes demographic, geographic, and behavioral dimensions, uncover hidden overlaps.
              You can use insights to create compelling "personas" or narratives. Identify archetypes or personas (e.g., "The Budget Shopper" or "The Early Adopter") using clustering
              and/or other apt ways.\n'''
        "5. Evaluate potential biases in data and explain how they manifest (e.g., gender or age biases in recruitment datasets). \n"
        "6. Reverse engineer hypotheses for the given data (e.g., Why did sales in region X drop?) and then generate plausible explanations using surrounding data. \n"
        "7. Map the dataset to emotional stories (e.g., in sales data, highlighting trends of success/failure as ups and downs in a business journey).\n"
        "Provide your analysis in a clear, structured manner in the form of a story by talking about: ."
        '''The data received, briefly
          The analysis you carried out
          The insights you discovered
          The implications of your findings (i.e. what to do with the insights)'''
    )

    # Make the API call using the make_api_call function
    response = make_api_call(prompt)

    return response



def generate_image_markdown(image_folder):
    # Get the list of PNG files in the image folder
    png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Generate the Markdown image links
    image_markdowns = []
    for image in png_files:
        image_path = os.path.join(image_folder, image)
        alt_text = image.split('.')[0]  # Use the file name (without extension) as alt text
        # Create relative paths for markdown
        markdown_link = f'![{alt_text}]({image})'  # Remove 'image_folder' to use relative path
        image_markdowns.append(markdown_link)

    return '\n'.join(image_markdowns)

def write_to_readme(readme_file, image_markdowns):
    # Open the README file and append the image Markdown
    with open(readme_file, 'a') as f: 
        f.write('\n\n' + image_markdowns)

#-----------------------------------------------------------------------------------------------------------

# Ensure the script is provided with a dataset argument
if len(sys.argv) < 2:
    print("Expected format: uv run code.py <dataset.csv>")
    sys.exit(1)

# Get the dataset path from the command-line argument
dataset_path = sys.argv[1]

# Extract the dataset name (without extension) to create a folder
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
output_folder = os.path.join(os.getcwd(), dataset_name)

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print(f"Output folder created: {output_folder}")

dataset = f"{dataset_name}.csv"
df = load_csv(dataset)
classifications = classify_columns(preprocess_dataframe(df))

# Perform analysis based on classification
for data_type, columns in classifications.items():
  if data_type == "Numerical":
    analyze_numerical(df, columns, output_folder)
    plot_numerical_distributions(df, columns, output_folder)
  elif data_type in ["Categorical", "Geographic"]:
    analyze_categorical_or_geographic(df, columns, output_folder)
  elif data_type == "Time Series":
    analyze_time_series(df, classifications["Time Series"], classifications["Numerical"], output_folder)

# Define the full file path
file_path = os.path.join(output_folder, 'README.md')

# Write the analysis of the data into the README.md file
with open(file_path, 'w') as f:
    f.write('''The code begins by loading a dataset, carefully selecting a sample to assess the most relevant relationships between variables. Using this sample, an API call is made to an advanced language model, which identifies key variables that can be further explored through correlation heatmaps, clustering, or time series analysis. With these insights, the data is preprocessed and categorized—distinguishing numerical, categorical, geographic, and time-series columns. Statistical techniques like PCA are applied for dimensionality reduction, and KMeans clustering uncovers hidden patterns. The final result is a series of visualizations—heatmaps, clustering plots, and time series graphs—offering a detailed understanding of the dataset's underlying structure.''')
    f.write("\n")
    f.write(generate_llm_analysis(df))
# Writing image markdown to README.md
    write_to_readme("README.md", generate_image_markdown(output_folder))
    
print(f"README.md file written to {file_path}")