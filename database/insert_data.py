"""
Data insertion script for Bank Reviews Analysis.

This script loads the analyzed reviews from CSV and inserts them into the
PostgreSQL bank_reviews database with proper bank ID mapping.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_insertion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ReviewDataInserter:
    """
    Handles insertion of review data from CSV into PostgreSQL database.
    """
    
    def __init__(self, csv_path, batch_size=100):
        """
        Initialize the data inserter.
        
        Args:
            csv_path (str): Path to the analyzed reviews CSV file
            batch_size (int): Number of rows to insert per batch
        """
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.db = DatabaseManager()
        self.bank_id_map = {}
        
        logger.info(f"ReviewDataInserter initialized with batch_size={batch_size}")
    
    def load_bank_mapping(self):
        """
        Load bank names to bank_id mapping from database.
        
        Returns:
            dict: Mapping of bank_name to bank_id
        """
        try:
            query = "SELECT bank_id, bank_name FROM banks;"
            results = self.db.execute_query(query)
            
            self.bank_id_map = {row[1]: row[0] for row in results}
            logger.info(f"Loaded bank mapping: {self.bank_id_map}")
            return self.bank_id_map
            
        except Exception as e:
            logger.error(f"Error loading bank mapping: {e}")
            raise
    
    def load_csv_data(self):
        """
        Load and validate the CSV data.
        
        Returns:
            pd.DataFrame: Loaded review data
        """
        try:
            logger.info(f"Loading CSV from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            logger.info(f"Loaded {len(df)} rows from CSV")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Validate required columns
            required_columns = ['review', 'rating', 'date', 'bank']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def prepare_data_for_insertion(self, df):
        """
        Prepare DataFrame for database insertion.
        
        Args:
            df (pd.DataFrame): Raw DataFrame from CSV
        
        Returns:
            list: List of tuples ready for database insertion
        """
        logger.info("Preparing data for insertion...")
        
        data_list = []
        skipped = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing data"):
            try:
                # Map bank name to bank_id
                bank_name = row['bank']
                if bank_name not in self.bank_id_map:
                    logger.warning(f"Unknown bank: {bank_name} at row {idx}")
                    skipped += 1
                    continue
                
                bank_id = self.bank_id_map[bank_name]
                
                # Parse date
                review_date = pd.to_datetime(row['date']).date()
                
                # Prepare tuple for insertion
                data_tuple = (
                    bank_id,
                    row['review'],
                    int(row['rating']),
                    review_date,
                    row.get('sentiment_label'),
                    float(row['sentiment_score']) if pd.notna(row.get('sentiment_score')) else None,
                    float(row['pos_score']) if pd.notna(row.get('pos_score')) else None,
                    float(row['neu_score']) if pd.notna(row.get('neu_score')) else None,
                    float(row['neg_score']) if pd.notna(row.get('neg_score')) else None,
                    bool(row.get('rating_adjusted', False)) if pd.notna(row.get('rating_adjusted')) else False,
                    row.get('themes_str'),  # identified_themes
                    row.get('preprocessed_text'),
                    row.get('source', 'Google Play')
                )
                
                data_list.append(data_tuple)
                
            except Exception as e:
                logger.warning(f"Error preparing row {idx}: {e}")
                skipped += 1
                continue
        
        logger.info(f"Prepared {len(data_list)} rows for insertion ({skipped} skipped)")
        return data_list
    
    def insert_data(self, data_list):
        """
        Insert data into the reviews table in batches.
        
        Args:
            data_list (list): List of tuples to insert
        
        Returns:
            int: Total number of rows inserted
        """
        insert_query = """
            INSERT INTO reviews (
                bank_id, review_text, rating, review_date,
                sentiment_label, sentiment_score,
                pos_score, neu_score, neg_score,
                rating_adjusted, identified_themes,
                preprocessed_text, source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        total_inserted = 0
        
        try:
            # Insert in batches
            for i in tqdm(range(0, len(data_list), self.batch_size), desc="Inserting batches"):
                batch = data_list[i:i + self.batch_size]
                rows_affected = self.db.execute_many(insert_query, batch)
                total_inserted += rows_affected
            
            logger.info(f"Successfully inserted {total_inserted} rows")
            return total_inserted
            
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            raise
    
    def verify_insertion(self):
        """
        Verify the inserted data by running summary queries.
        
        Returns:
            dict: Summary statistics
        """
        try:
            # Total count
            total_query = "SELECT COUNT(*) FROM reviews;"
            total = self.db.execute_query(total_query)[0][0]
            
            # Count by bank
            bank_query = """
                SELECT b.bank_name, COUNT(r.review_id) as count
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name;
            """
            bank_counts = self.db.execute_query(bank_query)
            
            # Average rating
            rating_query = "SELECT AVG(rating) FROM reviews;"
            avg_rating = self.db.execute_query(rating_query)[0][0]
            
            # Sentiment distribution
            sentiment_query = """
                SELECT sentiment_label, COUNT(*) as count
                FROM reviews
                WHERE sentiment_label IS NOT NULL
                GROUP BY sentiment_label;
            """
            sentiment_dist = self.db.execute_query(sentiment_query)
            
            summary = {
                'total_reviews': total,
                'reviews_by_bank': {row[0]: row[1] for row in bank_counts},
                'average_rating': float(avg_rating) if avg_rating else 0,
                'sentiment_distribution': {row[0]: row[1] for row in sentiment_dist}
            }
            
            logger.info("Verification complete:")
            logger.info(f"  Total reviews: {summary['total_reviews']}")
            logger.info(f"  By bank: {summary['reviews_by_bank']}")
            logger.info(f"  Avg rating: {summary['average_rating']:.2f}")
            logger.info(f"  Sentiment: {summary['sentiment_distribution']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            raise
    
    def run(self):
        """
        Execute the full data insertion pipeline.
        
        Returns:
            dict: Summary of insertion results
        """
        logger.info("="*60)
        logger.info("Starting Data Insertion Pipeline")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load bank mapping
            logger.info("\n[1/5] Loading bank mapping...")
            self.load_bank_mapping()
            
            # Step 2: Load CSV data
            logger.info("\n[2/5] Loading CSV data...")
            df = self.load_csv_data()
            
            # Step 3: Prepare data
            logger.info("\n[3/5] Preparing data for insertion...")
            data_list = self.prepare_data_for_insertion(df)
            
            if len(data_list) == 0:
                raise ValueError("No valid data to insert")
            
            # Step 4: Insert data
            logger.info(f"\n[4/5] Inserting {len(data_list)} reviews in batches of {self.batch_size}...")
            total_inserted = self.insert_data(data_list)
            
            # Step 5: Verify insertion
            logger.info("\n[5/5] Verifying insertion...")
            summary = self.verify_insertion()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "="*60)
            logger.info("Data Insertion Complete!")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Rows inserted: {total_inserted}")
            logger.info(f"Insertion rate: {total_inserted/duration:.2f} rows/second")
            logger.info("="*60)
            
            return summary
            
        except Exception as e:
            logger.error(f"Data insertion failed: {e}")
            raise


def main():
    """Main function to run the data insertion."""
    
    # Default path to analyzed reviews
    default_csv_path = "data/processed/analyzed_reviews.csv"
    
    # Allow command-line argument for CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv_path
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print(f"Usage: python database/insert_data.py [path_to_csv]")
        sys.exit(1)
    
    print(f"Inserting data from: {csv_path}")
    print("-" * 60)
    
    # Create inserter and run
    inserter = ReviewDataInserter(csv_path, batch_size=100)
    
    try:
        summary = inserter.run()
        
        print("\n" + "="*60)
        print("INSERTION SUMMARY")
        print("="*60)
        print(f"Total Reviews: {summary['total_reviews']}")
        print(f"\nReviews by Bank:")
        for bank, count in summary['reviews_by_bank'].items():
            print(f"  {bank}: {count}")
        print(f"\nAverage Rating: {summary['average_rating']:.2f}")
        print(f"\nSentiment Distribution:")
        for sentiment, count in summary['sentiment_distribution'].items():
            print(f"  {sentiment}: {count}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
