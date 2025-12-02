# Bank Reviews Database Documentation

## Overview

The `bank_reviews` database stores scraped Google Play Store reviews for three Ethiopian banks, along with sentiment analysis and thematic classification results. This database supports the Fintech App Review Analysis project.

**Database Name:** `bank_reviews`  
**DBMS:** PostgreSQL 14.19  
**Created:** December 1, 2025

---

## Database Credentials

Database credentials are stored securely in the `.env` file (not committed to version control).

**Default Configuration:**
- **Host:** `localhost`
- **Port:** `5432`
- **Database:** `bank_reviews`
- **User:** `analyst`
- **Password:** (stored in `.env`)

---

## Entity-Relationship Diagram

```
┌─────────────────────────┐
│       banks             │
├─────────────────────────┤
│ PK  bank_id (SERIAL)    │
│     bank_name (VARCHAR) │
│     app_name (VARCHAR)  │
│     app_id (VARCHAR)    │
│     created_at (TIMESTAMP)
└─────────────────────────┘
            │
            │ 1
            │
            │
            │ N
            ▼
┌─────────────────────────────────────┐
│           reviews                   │
├─────────────────────────────────────┤
│ PK  review_id (SERIAL)              │
│ FK  bank_id (INTEGER)               │
│     review_text (TEXT)              │
│     rating (INTEGER)                │
│     review_date (DATE)              │
│     sentiment_label (VARCHAR)       │
│     sentiment_score (DECIMAL)       │
│     pos_score (DECIMAL)             │
│     neu_score (DECIMAL)             │
│     neg_score (DECIMAL)             │
│     rating_adjusted (BOOLEAN)       │
│     identified_themes (TEXT)        │
│     preprocessed_text (TEXT)        │
│     source (VARCHAR)                │
│     created_at (TIMESTAMP)          │
└─────────────────────────────────────┘
```

**Relationship:** One bank can have many reviews (1:N)

---

## Table Schemas

### 1. `banks` Table

Stores information about the three Ethiopian banks and their mobile banking apps.

| Column       | Type         | Constraints      | Description                               |
| ------------ | ------------ | ---------------- | ----------------------------------------- |
| `bank_id`    | SERIAL       | PRIMARY KEY      | Auto-incrementing unique identifier       |
| `bank_name`  | VARCHAR(100) | NOT NULL, UNIQUE | Short name of the bank (CBE, BOA, Dashen) |
| `app_name`   | VARCHAR(200) | NOT NULL         | Full name of the mobile banking app       |
| `app_id`     | VARCHAR(200) | NOT NULL, UNIQUE | Google Play Store app package ID          |
| `created_at` | TIMESTAMP    | DEFAULT NOW()    | Record creation timestamp                 |

**Sample Data:**
```sql
 bank_id | bank_name |              app_name              |            app_id            
---------+-----------+------------------------------------+------------------------------
       1 | CBE       | Commercial Bank of Ethiopia Mobile | com.combanketh.mobilebanking
       2 | BOA       | Bank of Abyssinia                  | com.boa.boaMobileBanking    
       3 | Dashen    | Dashen Bank                        | com.cr2.amolelight          
```

---

### 2. `reviews` Table

Stores scraped reviews with sentiment analysis and thematic classification results.

| Column              | Type         | Constraints                   | Description                                  |
| ------------------- | ------------ | ----------------------------- | -------------------------------------------- |
| `review_id`         | SERIAL       | PRIMARY KEY                   | Auto-incrementing unique identifier          |
| `bank_id`           | INTEGER      | NOT NULL, FK → banks(bank_id) | Reference to the bank                        |
| `review_text`       | TEXT         | NOT NULL                      | Original review text from Google Play Store  |
| `rating`            | INTEGER      | NOT NULL, CHECK (1-5)         | User rating (1-5 stars)                      |
| `review_date`       | DATE         | NOT NULL                      | Date when the review was posted              |
| `sentiment_label`   | VARCHAR(20)  | NULL                          | Sentiment: Positive, Negative, or Neutral    |
| `sentiment_score`   | DECIMAL(5,4) | NULL                          | VADER compound sentiment score (-1 to 1)     |
| `pos_score`         | DECIMAL(5,4) | NULL                          | VADER positive sentiment component           |
| `neu_score`         | DECIMAL(5,4) | NULL                          | VADER neutral sentiment component            |
| `neg_score`         | DECIMAL(5,4) | NULL                          | VADER negative sentiment component           |
| `rating_adjusted`   | BOOLEAN      | DEFAULT FALSE                 | Flag: sentiment was adjusted based on rating |
| `identified_themes` | TEXT         | NULL                          | Comma-separated list of themes               |
| `preprocessed_text` | TEXT         | NULL                          | Cleaned text for NLP analysis                |
| `source`            | VARCHAR(50)  | DEFAULT 'Google Play'         | Data source                                  |
| `created_at`        | TIMESTAMP    | DEFAULT NOW()                 | Record insertion timestamp                   |

**Constraints:**
- `rating` must be between 1 and 5
- `bank_id` references `banks(bank_id)` with CASCADE delete

---

## Indexes

Performance indexes have been created on frequently queried columns:

| Index Name                | Column(s)         | Type   | Purpose                              |
| ------------------------- | ----------------- | ------ | ------------------------------------ |
| `idx_reviews_bank_id`     | `bank_id`         | B-tree | Fast joins with banks table          |
| `idx_reviews_rating`      | `rating`          | B-tree | Filter reviews by rating             |
| `idx_reviews_sentiment`   | `sentiment_label` | B-tree | Filter reviews by sentiment          |
| `idx_reviews_date`        | `review_date`     | B-tree | Time-series analysis                 |
| `idx_reviews_text_search` | `review_text`     | GIN    | Full-text search for keyword queries |

---

## Setup Instructions

### 1. Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Verify installation
psql --version
```

### 2. Start PostgreSQL Service

```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql
systemctl is-active postgresql
```

### 3. Create Database and User

```bash
# Create database
sudo -u postgres psql -c "CREATE DATABASE bank_reviews;"

# Create user with password
sudo -u postgres psql -c "CREATE USER analyst WITH PASSWORD 'your_password';"

# Grant privileges
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE bank_reviews TO analyst;"
```

### 4. Run Schema Creation Script

```bash
# Execute schema.sql to create tables
sudo -u postgres psql -d bank_reviews < database/schema.sql

# Verify tables were created
sudo -u postgres psql -d bank_reviews -c "\dt"
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bank_reviews
DB_USER=analyst
DB_PASSWORD=your_password
```

---

## Connection Examples

### Python (psycopg2)

```python
import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM banks;")
print(cursor.fetchall())
cursor.close()
conn.close()
```

### Python (pandas)

```python
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

# Read data into DataFrame
df = pd.read_sql("SELECT * FROM reviews WHERE rating = 5;", conn)
print(df.head())
conn.close()
```

### Command Line (psql)

```bash
# Connect as analyst user
psql -h localhost -U analyst -d bank_reviews

# Connect as postgres superuser
sudo -u postgres psql -d bank_reviews

# Run a query directly
psql -h localhost -U analyst -d bank_reviews -c "SELECT COUNT(*) FROM reviews;"
```

---

## Sample Queries

### Count Reviews per Bank

```sql
SELECT b.bank_name, COUNT(r.review_id) AS total_reviews
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name;
```

### Average Rating per Bank

```sql
SELECT b.bank_name, ROUND(AVG(r.rating), 2) AS avg_rating
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY avg_rating DESC;
```

### Sentiment Distribution per Bank

```sql
SELECT 
    b.bank_name,
    r.sentiment_label,
    COUNT(*) AS count
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
WHERE r.sentiment_label IS NOT NULL
GROUP BY b.bank_name, r.sentiment_label
ORDER BY b.bank_name, r.sentiment_label;
```

### Find Reviews Mentioning Specific Keywords

```sql
SELECT 
    b.bank_name,
    r.review_text,
    r.rating,
    r.sentiment_label
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
WHERE r.review_text ILIKE '%slow%' 
   OR r.review_text ILIKE '%crash%'
ORDER BY r.review_date DESC
LIMIT 10;
```

### Full-Text Search on Reviews

```sql
SELECT 
    b.bank_name,
    r.review_text,
    r.rating
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
WHERE to_tsvector('english', r.review_text) @@ to_tsquery('english', 'transfer & slow')
LIMIT 10;
```

### Reviews by Theme

```sql
SELECT 
    b.bank_name,
    r.identified_themes,
    COUNT(*) AS theme_count
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
WHERE r.identified_themes LIKE '%Technical Issues%'
GROUP BY b.bank_name, r.identified_themes;
```

### Time-Series Sentiment Analysis

```sql
SELECT 
    DATE_TRUNC('month', r.review_date) AS month,
    b.bank_name,
    ROUND(AVG(r.sentiment_score), 4) AS avg_sentiment
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
WHERE r.sentiment_score IS NOT NULL
GROUP BY month, b.bank_name
ORDER BY month, b.bank_name;
```

---

## Database Maintenance

### Backup Database

```bash
# Backup entire database
pg_dump -U analyst -d bank_reviews > backup_$(date +%Y%m%d).sql

# Backup schema only
pg_dump -U analyst -s bank_reviews > schema_backup.sql
```

### Restore Database

```bash
# Restore from backup
psql -U analyst -d bank_reviews < backup_20251201.sql
```

### View Database Size

```sql
SELECT pg_size_pretty(pg_database_size('bank_reviews'));
```

### View Table Sizes

```sql
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## Troubleshooting

### Permission Denied Errors

If you encounter "Permission denied" when running `sudo -u postgres` commands:
```bash
# This is normal - postgres user doesn't have access to your home directory
# The commands still execute successfully
```

### Connection Refused

```bash
# Check if PostgreSQL is running
systemctl status postgresql

# Start if not running
sudo systemctl start postgresql
```

### Authentication Failed

- Verify credentials in `.env` file
- Check PostgreSQL authentication config: `/etc/postgresql/14/main/pg_hba.conf`
- Try connecting as postgres superuser first

### Reset User Password

```bash
sudo -u postgres psql -c "ALTER USER analyst WITH PASSWORD 'new_password';"
```

---

## Next Steps

1. **Data Insertion**: Use Python scripts to load the cleaned and analyzed review data
2. **Query Optimization**: Monitor query performance and add indexes as needed
3. **Data Validation**: Run verification queries to ensure data integrity
4. **Backup Strategy**: Set up automated backups
5. **Documentation**: Keep this README updated as schema evolves

---

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/14/)
- [psycopg2 Documentation](https://www.psycopg.org/docs/)
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/14/textsearch.html)
