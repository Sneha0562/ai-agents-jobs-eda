# AI Agents Global Industry Jobs and Security - EDA

Comprehensive Exploratory Data Analysis (EDA) for the Kaggle dataset on AI Agents Global Industry Jobs and Security.

## Dataset

**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/zkskhurram/ai-agents-global-industry-jobs-and-security)

**Dataset Name:** `zkskhurram/ai-agents-global-industry-jobs-and-security`

This dataset provides insights into:
- Global AI agent job market trends
- Industry-wide security implications
- Job roles and requirements in the AI agents space

## Features

This EDA script provides comprehensive analysis including:

✅ **Dataset Overview**
- Shape, dimensions, and basic info
- Column information and data types
- First rows preview
- Basic statistical summary

✅ **Missing Values Analysis**
- Identifies missing data patterns
- Calculates missing percentages
- Visualizes missing values distribution

✅ **Data Types Analysis**
- Categorizes numerical vs categorical features
- Provides data type distribution breakdown

✅ **Numerical Features Analysis**
- Descriptive statistics
- Distribution histograms
- Min, max, mean, median, standard deviation

✅ **Categorical Features Analysis**
- Unique value counts
- Top values distribution
- Bar charts for categorical features

✅ **Correlation Analysis**
- Correlation matrix for numerical features
- Heatmap visualization
- Identifies feature relationships

✅ **Duplicate Detection**
- Identifies duplicate records
- Shows duplicate patterns

✅ **Outlier Detection**
- Uses IQR (Interquartile Range) method
- Identifies outliers in numerical columns
- Calculates outlier percentages

✅ **Summary Statistics**
- Comprehensive metrics table
- Memory usage analysis

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sneha0562/ai-agents-jobs-eda.git
cd ai-agents-jobs-eda
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Setup Kaggle API

1. Go to [Kaggle Settings](https://www.kaggle.com/settings/account)
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` file to:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **macOS/Linux:** `~/.kaggle/kaggle.json`

## Usage

Run the EDA script:

```bash
python eda_ai_agents_jobs.py
```

The script will:
1. Download the dataset from Kaggle
2. Load and analyze the data
3. Print detailed analysis to console
4. Generate 5 visualization PNG files:
   - `01_missing_values.png`
   - `02_data_types.png`
   - `03_numerical_distributions.png`
   - `04_categorical_distributions.png`
   - `05_correlation_heatmap.png`

## Output

The script generates:

### Console Output
- Detailed analysis of all aspects of the dataset
- Summary statistics and metrics
- Missing values information
- Outlier detection results
- Data quality insights

### Visualizations (PNG files)
1. **01_missing_values.png** - Bar chart showing missing value percentages
2. **02_data_types.png** - Distribution of data types in the dataset
3. **03_numerical_distributions.png** - Histograms for all numerical features
4. **04_categorical_distributions.png** - Bar charts for categorical features
5. **05_correlation_heatmap.png** - Correlation matrix heatmap

## Project Structure

```
ai-agents-jobs-eda/
├── eda_ai_agents_jobs.py      # Main EDA script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── *.png                       # Generated visualizations (after running)
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- kagglehub

See `requirements.txt` for specific versions.

## Key Insights from EDA

After running the script, you'll gain insights into:

- **Data Quality:** Missing values, duplicates, and data types
- **Feature Distributions:** How different features are distributed
- **Correlations:** Relationships between numerical features
- **Outliers:** Unusual data points that may need investigation
- **Data Characteristics:** Overall structure and composition

## Troubleshooting

### Kaggle API Error
```
Error: Could not find kagglehub credentials
```
**Solution:** Ensure your `kaggle.json` is in the correct location and has proper permissions.

### Import Error
```
ModuleNotFoundError: No module named 'kagglehub'
```
**Solution:** Run `pip install -r requirements.txt` to install all dependencies.

### Memory Issues
If you encounter memory issues with large datasets:
- Reduce the number of bins in histograms
- Process data in chunks
- Use a machine with more RAM

## Contributing

Feel free to:
- Fork this repository
- Add new analysis techniques
- Improve visualizations
- Submit pull requests

## License

This project is open source and available under the MIT License.

## Author

**Sneha0562** - GitHub

## Acknowledgments

- Dataset provided by [zkskhurram on Kaggle](https://www.kaggle.com/zkskhurram)
- Built with Python, Pandas, Matplotlib, and Seaborn

## References

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
