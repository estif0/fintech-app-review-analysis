# Fintech App Review Analysis 

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
- **Advanced Sentiment Analysis**: Hybrid VADER + rating-based approach with 96.8% accuracy
- **Thematic Classification**: Multi-label theme identification (8 themes, 84.8% coverage)
- **Text Processing Pipeline**: Complete NLP workflow with preprocessing, keyword extraction, and lemmatization
- **Language Processing**: Filters and processes English-language reviews for analysis
- **Quality Assurance**: Comprehensive data validation and quality checks
- **Modular Architecture**: Object-oriented design with reusable components
- **Extensive Testing**: 260+ unit tests ensuring code reliability
- **End-to-End Pipeline**: Integrated analysis workflow processing 827 reviews in ~5 seconds

## 🎓 Key Findings & Results

### Overall Sentiment Performance

| Bank                            | Positive Reviews | Avg Sentiment | Avg Rating | Market Position     |
| ------------------------------- | ---------------- | ------------- | ---------- | ------------------- |
| **Dashen Bank**                 | 69.7%            | 0.370         | 3.93       | 🥇 Leader            |
| **Commercial Bank of Ethiopia** | 64.8%            | 0.260         | 3.77       | 🥈 Strong            |
| **Bank of Abyssinia**           | 35.9%            | -0.005        | 2.64       | 🥉 Needs Improvement |

### Top Satisfaction Drivers (Per Bank)

**Dashen Bank** - Market Leader ⭐
- Outstanding user experience (74.5% of positive reviews)
- Fast performance and reliability (25.9%)
- Consistently praised as "super app"

**Commercial Bank of Ethiopia**
- Good user experience (59.2%)
- Appreciated features and functionality (25.2%)
- Strong digital service offerings

**Bank of Abyssinia**
- User-friendly interface when working (61.5%)
- Basic features appreciated (15.4%)
- Limited positive feedback

### Critical Pain Points Identified

**Bank of Abyssinia** - Urgent Action Required
- 🔴 Developer options bug (unique critical issue)
- 🔴 "Not working" technical failures (45.9%)
- 🔴 Performance problems (32.5%)

**Commercial Bank of Ethiopia**
- 🟡 Android missing iOS features (46.3%)
- 🟡 Branch verification requirement (24.1%)
- 🟡 Technical issues and crashes (37%)

**Dashen Bank** - Maintenance Focus
- 🟢 "Temporarily unavailable" errors (40.8%)
- 🟢 Performance degradation (39.4%)
- 🟢 Account opening issues (33.8%)

### Strategic Recommendations

Each bank received 9+ prioritized recommendations across:
- **Urgent (0-1 month)**: Critical bug fixes and stability
- **High Priority (1-3 months)**: Feature parity and performance
- **Medium Priority (3-6 months)**: Competitive enhancements
- **Quick Wins**: User-requested features (fingerprint auth, notifications)

## 📊 Data Collection

### Methodology

Reviews are collected using the `google-play-scraper` library with the following parameters:
- **Source**: Google Play Store (official API)
- **Target**: 500 reviews per bank
- **Language**: English reviews (filtered from multi-language dataset)
- **Sort Order**: Newest first
- **Date Range**: August 2024 - November 2025

### Statistics

| Metric                    | Value                    |
| ------------------------- | ------------------------ |
| Total Reviews Scraped     | 1,500                    |
| English Reviews (Cleaned) | 827                      |
| Banks Covered             | 3                        |
| Average Rating            | 3.43/5                   |
| Date Range                | 2024-08-01 to 2025-11-26 |
| Sentiment Accuracy        | 96.8%                    |
| Theme Coverage            | 84.8%                    |
| Database Records          | 827 (PostgreSQL)         |

**Reviews per Bank (English only):**
- Dashen Bank: 310 reviews (69.7% positive)
- Bank of Abyssinia: 290 reviews (35.9% positive)
- Commercial Bank of Ethiopia: 227 reviews (64.8% positive)

## 🔧 Data Preprocessing

### Cleaning Steps

1. **Duplicate Removal**: Identifies and removes duplicate reviews based on review ID, content, and timestamp
2. **Missing Value Handling**: 
   - Critical fields (review text, rating): Rows removed if missing
   - Non-critical fields (reply content, thumbs up): Filled with defaults
3. **Date Normalization**: Standardizes all dates to YYYY-MM-DD format
4. **Language Filtering**: Uses language detection to keep only English reviews
5. **Column Standardization**: Renames and organizes columns for consistency

### Data Quality

- **Missing Data**: <5% (primarily from optional reply fields)
- **Final Dataset**: 827 English reviews
- **Rating Distribution**:
  - 5 stars: 423 (51%)
  - 1 star: 251 (30%)
  - 2-4 stars: 153 (19%)

## 🧠 Sentiment & Thematic Analysis

### Sentiment Analysis (96.8% Accuracy)

We implement a **hybrid sentiment analysis approach** combining:

1. **VADER Lexicon-Based Analysis**: Industry-standard sentiment scoring
2. **Rating-Based Adjustments**: Corrects edge cases where sentiment contradicts rating
3. **Negative Pattern Detection**: 29 patterns to catch factual complaints ("not working", "crash", etc.)

**Why Hybrid?** Initial VADER-only approach had 50.6% accuracy for 1-star reviews. The hybrid approach improved this to **96.8%**.

**Results:**
- Overall Sentiment: 61.7% Positive, 36.8% Negative, 1.6% Neutral
- Low-rated reviews (1-2 stars): 95.9% correctly classified as negative
- High-rated reviews (4-5 stars): 97.3% correctly classified as positive
- Rating-adjusted reviews: 221 (26.7%)

### Thematic Classification (84.8% Coverage)

We identified **8 key themes** using rule-based multi-label classification:

1. **User Experience** - Ease of use, interface quality (good, best, great, easy, simple)
2. **Technical Issues** - Crashes, bugs, errors (crash, bug, error, not working, broken)
3. **Performance** - Speed and responsiveness (slow, fast, loading, lag, quick)
4. **Features & Functionality** - App capabilities (transfer, balance, service, feature)
5. **Updates & Improvements** - App changes (update, new, improve, change, upgrade)
6. **Authentication & Security** - Login and security (login, password, biometric, secure)
7. **Customer Support** - Service and help (support, help, customer, contact)
8. **Negative Experience** - Strong dissatisfaction (worst, terrible, bad, useless)

**Results:**
- Reviews with themes: 701/827 (84.8%)
- Average themes per review: 1.84
- Most common themes: User Experience, Technical Issues, Performance

### Text Preprocessing Pipeline

1. **Cleaning**: Lowercase, remove special chars, URLs, emails
2. **Tokenization**: Split into words using NLTK
3. **Stopword Removal**: 217 total (NLTK + custom banking terms)
4. **Lemmatization**: Reduce words to base forms (WordNet)

**Statistics:**
- Preprocessed reviews: 825/827
- Average tokens per review: 7.64
- Unique tokens: 1,738

### Keyword Extraction (TF-IDF)

- **Overall keywords**: Top 50 extracted (good, best, great, easy, fast)
- **Bank-specific keywords**: 30 per bank
- **N-grams**: 20 bigrams + 20 trigrams ("not working", "dashen super", etc.)
- **Features extracted**: 1,211 total

### Analysis Pipeline

The `AnalysisPipeline` orchestrates all analysis steps:
1. Load cleaned data → 2. Sentiment analysis → 3. Text preprocessing → 4. Keyword extraction → 5. Theme classification

**Performance:**
- Processing time: 4.95 seconds for 827 reviews
- Average: ~6ms per review
- Output: 14-column enriched dataset + 7 supporting files

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

### Complete Analysis Workflow

**Option 1: Use Demonstration Notebooks (Recommended)**

Explore the complete analysis with visualizations:
```bash
jupyter notebook notebooks/insights_recommendations.ipynb
```

This notebook includes:
- Executive summary with all key findings
- Data loading and statistical overview
- Satisfaction drivers analysis (2+ per bank)
- Pain points identification (2+ per bank)
- Business scenario investigations (retention, features, complaints)
- Actionable recommendations with priorities
- 5 professional visualizations
- Ethical considerations and limitations

**Option 2: Run Analysis Pipeline**

Execute the complete analysis programmatically:
```python
from core.analysis_pipeline import AnalysisPipeline

pipeline = AnalysisPipeline(
    output_dir='data/processed',
    use_rating_boost=True
)

results = pipeline.run(
    input_file='data/processed/cleaned_reviews.csv',
    output_file='data/processed/analyzed_reviews.csv',
    save_intermediate=True,
    show_progress=True
)
```

### Database Access

Query the PostgreSQL database with 827 reviews:
```bash
jupyter notebook notebooks/database_implementation.ipynb
```

Or connect directly:
```python
from database.db_connection import DatabaseManager

db = DatabaseManager()
reviews = db.execute_query("""
    SELECT b.bank_name, r.rating, r.sentiment_label, r.review_text
    FROM reviews r
    JOIN banks b ON r.bank_id = b.bank_id
    WHERE r.sentiment_label = 'Negative'
    ORDER BY r.review_date DESC
    LIMIT 10;
""")
```

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

### Running Sentiment & Thematic Analysis

Execute the complete analysis pipeline:
```bash
python core/analysis_pipeline.py
```

Or use the Python API:
```python
from core.analysis_pipeline import AnalysisPipeline

pipeline = AnalysisPipeline(
    output_dir='data/processed',
    use_rating_boost=True
)

results = pipeline.run(
    input_file='data/processed/cleaned_reviews.csv',
    output_file='data/processed/analyzed_reviews.csv',
    save_intermediate=True,
    show_progress=True
)
```

This generates:
- `analyzed_reviews.csv` - Enriched dataset with sentiment, themes, preprocessed text
- `keywords_overall.csv` - Top 50 keywords
- `keywords_by_bank.csv` - Bank-specific keywords
- `bigrams.csv` & `trigrams.csv` - N-gram phrases
- Intermediate files for each analysis step

### Exploring Results

Open the demonstration notebooks:

**Sentiment & Thematic Analysis:**
```bash
jupyter notebook notebooks/sentiment_thematic_analysis.ipynb
```

**Insights & Recommendations:**
```bash
jupyter notebook notebooks/insights_recommendations.ipynb
```

**Database Exploration:**
```bash
jupyter notebook notebooks/database_implementation.ipynb
```

The notebooks include:
- Complete analysis demonstrations
- Data visualizations (5+ plots)
- Validation checks and statistics
- Executive summaries
- Business recommendations

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
pytest tests/test_sentiment_analyzer.py -v
pytest tests/test_text_preprocessor.py -v
pytest tests/test_keyword_extractor.py -v
pytest tests/test_theme_classifier.py -v
```

All 260 tests should pass ✅

## 📁 Project Structure

```
fintech-app-review-analysis/
├── core/                          # Core modules (OOP design)
│   ├── config.py                 # Configuration management
│   ├── scraper.py                # Review scraping logic
│   ├── preprocessor.py           # Data preprocessing pipeline
│   ├── sentiment_analyzer.py     # Hybrid sentiment analysis (96.8% accuracy)
│   ├── text_preprocessor.py      # NLP text preprocessing
│   ├── keyword_extractor.py      # TF-IDF keyword extraction
│   ├── theme_classifier.py       # Multi-label theme classification
│   └── analysis_pipeline.py      # End-to-end analysis orchestrator
├── database/                      # PostgreSQL database
│   ├── db_connection.py          # DatabaseManager class
│   ├── insert_data.py            # Data insertion utilities
│   ├── schema.sql                # Database schema definition
│   ├── verification_queries.sql  # 16 validation queries
│   ├── schema_dump.sql           # Schema backup
│   ├── full_dump.sql             # Complete database backup
│   └── README.md                 # Database documentation
├── notebooks/                     # Jupyter notebooks (demonstrations)
│   ├── exploratory_data_analysis.ipynb
│   ├── sentiment_thematic_analysis.ipynb  # Task 2 demonstration
│   ├── database_implementation.ipynb      # Task 3 demonstration
│   └── insights_recommendations.ipynb     # Task 4 complete analysis
├── scripts/                       # Executable scripts
│   └── scrape_all_banks.py       # Batch scraping utility
├── tests/                         # Unit tests (260 tests, 100% passing)
│   ├── test_config.py            # 22 tests
│   ├── test_scraper.py           # 22 tests
│   ├── test_preprocessor.py      # 31 tests
│   ├── test_sentiment_analyzer.py  # 43 tests
│   ├── test_text_preprocessor.py   # 48 tests
│   ├── test_keyword_extractor.py   # 45 tests
│   └── test_theme_classifier.py    # 48 tests
├── data/                          # Data storage (gitignored)
│   ├── raw/                      # Raw scraped data (1,500 reviews)
│   │   ├── boa_reviews_raw.csv
│   │   ├── cbe_reviews_raw.csv
│   │   ├── dashen_reviews_raw.csv
│   │   └── scraping_summary.json
│   └── processed/                # Cleaned and analyzed data
│       ├── cleaned_reviews.csv           # 827 English reviews
│       ├── analyzed_reviews.csv          # Enriched with sentiment & themes
│       ├── sentiment_analyzed_reviews.csv
│       ├── preprocessed_reviews.csv
│       ├── themed_reviews.csv
│       ├── keywords_overall.csv          # Top 50 keywords
│       ├── keywords_by_bank.csv          # Bank-specific keywords
│       ├── bigrams.csv                   # 20 top bigrams
│       ├── trigrams.csv                  # 20 top trigrams
│       ├── sentiment_statistics.csv
│       ├── bank_comparison_stats.csv
│       ├── eda_summary_statistics.csv
│       └── data_quality_report.txt
├── reports/                       # Analysis reports and visualizations
│   └── figures/                  # Publication-ready plots (5 visualizations)
│       ├── rating_distribution.png
│       ├── sentiment_distribution.png
│       ├── average_sentiment.png
│       ├── themes_distribution.png
│       └── sentiment_rating_correlation.png
├── .github/workflows/            # CI/CD pipelines
│   └── ci.yml                    # Continuous integration
│   └── unittest.yml              # Automated testing
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🧪 Testing

The project includes comprehensive unit tests (260 total):

**Task 1 - Data Collection:**
- **Configuration Tests** (22 tests): Validate settings and helper methods
- **Scraper Tests** (22 tests): Test data collection with mocked API calls
- **Preprocessor Tests** (31 tests): Verify cleaning and filtering logic

**Task 2 - Sentiment & Thematic Analysis:**
- **Sentiment Analyzer Tests** (43 tests): Hybrid approach, pattern detection, rating adjustments
- **Text Preprocessor Tests** (48 tests): Cleaning, tokenization, lemmatization, stopwords
- **Keyword Extractor Tests** (45 tests): TF-IDF, n-grams, bank-specific keywords
- **Theme Classifier Tests** (48 tests): Multi-label classification, theme statistics

**Coverage:**
- All 260 tests passing ✅
- End-to-end integration validated
- CI/CD pipeline with GitHub Actions

All tests follow OOP principles with proper fixtures and mocking.

## 🔒 Known Limitations

1. **Selection Bias**: Only Google Play Store (Android) users are represented
2. **Negativity Bias**: Unhappy users more likely to leave reviews (U-shaped distribution)
3. **Language Bias**: Only English reviews analyzed (53% of original dataset filtered out)
4. **Recency Bias**: 86% of reviews from 2025, may not reflect long-term trends
5. **Sample Size**: Smaller sample for CBE (227) vs BOA (290) and Dashen (310)
6. **Temporal Coverage**: Limited to 16 months (Aug 2024 - Nov 2025)
7. **Theme Classification**: Rule-based approach may miss nuanced or emerging themes
8. **Sentiment Edge Cases**: Some sarcasm or context-dependent meaning may be missed
9. **API Rate Limits**: Google Play Store scraping is throttled to respect service limits

### Mitigation Strategies

For future iterations:
- Expand to iOS App Store reviews
- Include multi-language analysis (Amharic, Oromo)
- Conduct quarterly analysis to track trends
- Complement with user surveys for balanced feedback
- Implement machine learning for dynamic theme discovery
- Validate with A/B testing of recommendations

## 📚 Documentation

### Complete Analysis Report

The full analysis with findings and recommendations is available in:
- **Notebook**: `notebooks/insights_recommendations.ipynb` (interactive)
- **Report**: `reports/final_report.pdf` (10-page summary)

### Technical Documentation

- **Database Schema**: `database/README.md`
- **Progress Tracker**: `docs/steps.md`
- **API Documentation**: Inline docstrings in all modules

### Reproducing the Analysis

1. **Data Collection**:
   ```bash
   python scripts/scrape_all_banks.py --count 500
   python core/preprocessor.py
   ```

2. **Analysis Pipeline**:
   ```bash
   python core/analysis_pipeline.py
   ```

3. **Database Setup**:
   ```bash
   psql -U postgres < database/schema.sql
   python database/insert_data.py
   ```

4. **Explore Results**:
   ```bash
   jupyter notebook notebooks/insights_recommendations.ipynb
   ```

All outputs will be generated in `data/processed/` and `reports/figures/`.

## 📝 Development Guidelines

- **Code Style**: Follow PEP 8 guidelines
- **Architecture**: Object-oriented design with comprehensive docstrings
- **Testing**: Write tests for all new features (target: 100% test pass rate)
- **Documentation**: Update README and inline comments
- **Type Hints**: Use type annotations for better code clarity
- **Git Workflow**: Feature branches → Pull requests → Main
- **Commit Messages**: Follow conventional commits (feat:, fix:, docs:, etc.)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Estifanos**
- GitHub: [@estif0](https://github.com/estif0)

## 🙏 Acknowledgments

- **10 Academy AI Mastery Program** for project framework and guidance
- **Google Play Scraper** library for review collection capabilities
- **VADER Sentiment Analysis** for robust lexicon-based sentiment scoring
- **NLTK & scikit-learn** for NLP and machine learning utilities
- **PostgreSQL** for reliable data storage and querying
- **Ethiopian Banking Sector** for providing innovative mobile banking services
- **Open Source Community** for the excellent Python ecosystem

## 📞 Contact & Support

- **Author**: Estifanos
- **GitHub**: [@estif0](https://github.com/estif0)
- **Repository**: [fintech-app-review-analysis](https://github.com/estif0/fintech-app-review-analysis)
- **Issues**: Please report bugs or request features via GitHub Issues

## 🎯 Project Deliverables

- ✅ **Data Collection**: 1,500 reviews scraped → 827 English reviews
- ✅ **Sentiment Analysis**: 96.8% accuracy with hybrid approach
- ✅ **Thematic Classification**: 8 themes, 84.8% coverage
- ✅ **PostgreSQL Database**: 827 records with full schema
- ✅ **Visualizations**: 5 publication-ready plots
- ✅ **Recommendations**: 9+ actionable items per bank
- ✅ **Testing**: 260 unit tests, 100% passing
- ✅ **Documentation**: Complete notebooks and technical docs

## 📅 Project Timeline

- **Phase 1 (Task 1)**: Data Collection & Preprocessing ✓
- **Phase 2 (Task 2)**: Sentiment & Thematic Analysis ✓
- **Phase 3 (Task 3)**: PostgreSQL Database Implementation ✓
- **Phase 4 (Task 4)**: Insights & Recommendations ✓

**Final Submission**: December 2, 2025

---

**Note**: This project is for educational and research purposes. All data is publicly available from Google Play Store. Analysis and recommendations are evidence-based suggestions intended to improve customer satisfaction.
