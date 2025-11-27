# Ethiopian Banking Apps: Customer Sentiment Analysis

A comprehensive analysis of customer reviews for Ethiopian mobile banking applications, focusing on sentiment analysis and thematic insights to improve customer satisfaction.

## 📋 Project Overview

This project analyzes customer reviews from Google Play Store for three major Ethiopian banks:
- **Commercial Bank of Ethiopia (CBE)** - Mobile Banking
- **Bank of Abyssinia (BOA)** - Mobile Banking  
- **Dashen Bank** - Amole Light Mobile Banking

The analysis aims to:
1. Understand customer sentiment and satisfaction levels
2. Identify common themes and pain points
3. Provide actionable recommendations for service improvement
4. Track sentiment trends over time

## 🎯 Key Features

- **Automated Data Collection**: Scrapes reviews from Google Play Store using official APIs
- **Language Processing**: Filters and processes English-language reviews for sentiment analysis
- **Quality Assurance**: Comprehensive data validation and quality checks
- **Modular Architecture**: Object-oriented design with reusable components
- **Extensive Testing**: 76+ unit tests ensuring code reliability

## 📊 Data Collection

### Methodology

Reviews are collected using the `google-play-scraper` library with the following parameters:
- **Source**: Google Play Store (official API)
- **Target**: 500 reviews per bank
- **Language**: English reviews (filtered from multi-language dataset)
- **Sort Order**: Newest first
- **Date Range**: July 2022 - November 2025

### Statistics

| Metric | Value |
|--------|-------|
| Total Reviews Scraped | 1,500 |
| English Reviews (Cleaned) | 797 |
| Banks Covered | 3 |
| Average Rating | 3.40/5 |
| Date Range | 2022-07-16 to 2025-11-26 |

**Reviews per Bank (English only):**
- Commercial Bank of Ethiopia: 231 reviews
- Bank of Abyssinia: 288 reviews
- Dashen Bank: 278 reviews

## 🔧 Data Preprocessing

### Cleaning Steps

1. **Duplicate Removal**: Identifies and removes duplicate reviews based on review ID, content, and timestamp
2. **Missing Value Handling**: 
   - Critical fields (review text, rating): Rows removed if missing
   - Non-critical fields (reply content, thumbs up): Filled with defaults
3. **Date Normalization**: Standardizes all dates to YYYY-MM-DD format
4. **Language Filtering**: Uses language detection to keep only English reviews (53% of total)
5. **Column Standardization**: Renames and organizes columns for consistency

### Data Quality

- **Missing Data**: 5.55% (primarily from optional reply fields)
- **Language Distribution** (before filtering):
  - English: 797 (53%)
  - Somali: 171 (11%)
  - Unknown: 115 (8%)
  - Other languages: 417 (28%)
- **Rating Distribution**:
  - 5 stars: 387 (49%)
  - 1 star: 240 (30%)
  - 2-4 stars: 170 (21%)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/estif0/fintech-app-review-analysis.git
cd fintech-app-review-analysis
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Scraping Reviews

Scrape reviews from all three banks:
```bash
python scripts/scrape_all_banks.py --count 500
```

Options:
- `--count`: Number of reviews per bank (default: 500)
- `--separate`: Save separate CSV files per bank (default: True)
- `--combined`: Also save combined CSV file (default: False)

### Preprocessing Data

Clean and filter the scraped reviews:
```bash
python core/preprocessor.py
```

This will:
- Load raw reviews from `data/raw/`
- Remove duplicates and handle missing values
- Filter for English-language reviews only
- Save cleaned data to `data/processed/cleaned_reviews.csv`
- Generate quality report in `data/processed/data_quality_report.txt`

### Running Tests

Execute the test suite:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_config.py -v
pytest tests/test_scraper.py -v
pytest tests/test_preprocessor.py -v
```

## 📁 Project Structure

```
fintech-app-review-analysis/
├── core/                      # Core modules
│   ├── config.py             # Configuration management
│   ├── scraper.py            # Review scraping logic
│   └── preprocessor.py       # Data preprocessing pipeline
├── scripts/                   # Executable scripts
│   └── scrape_all_banks.py   # Batch scraping utility
├── tests/                     # Unit tests
│   ├── test_config.py
│   ├── test_scraper.py
│   └── test_preprocessor.py
├── data/                      # Data storage (gitignored)
│   ├── raw/                  # Raw scraped data
│   └── processed/            # Cleaned data
├── database/                  # Database schemas (future)
├── reports/                   # Analysis reports
└── requirements.txt          # Python dependencies
```

## 🧪 Testing

The project includes comprehensive unit tests:

- **Configuration Tests** (22 tests): Validate settings and helper methods
- **Scraper Tests** (22 tests): Test data collection with mocked API calls
- **Preprocessor Tests** (31 tests): Verify cleaning and filtering logic
- **Integration Tests**: End-to-end workflow validation

All tests follow OOP principles with proper fixtures and mocking.

## 🔒 Known Limitations

1. **Language Detection Accuracy**: Short reviews (< 3 words) may be misclassified
2. **Date Range**: Limited to publicly available reviews on Google Play Store
3. **Sample Size**: English reviews represent ~50% of total due to multilingual user base
4. **API Rate Limits**: Google Play Store scraping is throttled to respect service limits
5. **Missing Replies**: 99.87% of reviews lack bank responses in the data

## 📝 Development Guidelines

- **Code Style**: Follow PEP 8 guidelines
- **Architecture**: Object-oriented design with comprehensive docstrings
- **Testing**: Write tests for all new features
- **Documentation**: Update README and inline comments
- **Type Hints**: Use type annotations for better code clarity

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Estifanos**
- GitHub: [@estif0](https://github.com/estif0)

## 🙏 Acknowledgments

- Google Play Scraper library for review collection
- 10 Academy AI Mastery Program for project framework
- Ethiopian banking sector for providing mobile banking services

---

**Note**: This project is for educational and research purposes. All data is publicly available from Google Play Store.
