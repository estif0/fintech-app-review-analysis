"""
Script to scrape reviews for all Ethiopian bank mobile apps.

This script collects reviews from Google Play Store for CBE, BOA, and Dashen Bank
apps, saves them to CSV files, and generates a summary report.

Usage:
    python scripts/scrape_all_banks.py --count 400
    python scripts/scrape_all_banks.py --count 500 --output-dir data/raw
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scraper import ReviewScraper, ReviewScraperError
from core.config import Config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Scrape reviews for Ethiopian bank mobile apps from Google Play Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape 400 reviews per bank (default)
  python scripts/scrape_all_banks.py
  
  # Scrape 500 reviews per bank
  python scripts/scrape_all_banks.py --count 500
  
  # Specify custom output directory
  python scripts/scrape_all_banks.py --count 400 --output-dir data/custom
  
  # Save summary to specific file
  python scripts/scrape_all_banks.py --summary-file scraping_report.json
        """,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=400,
        help="Number of reviews to scrape per bank (default: 400)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV files (default: data/raw/)",
    )

    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Path to save scraping summary JSON (default: data/raw/scraping_summary.json)",
    )

    parser.add_argument(
        "--separate-files",
        action="store_true",
        default=True,
        help="Save each bank to separate CSV files (default: True)",
    )

    parser.add_argument(
        "--combined-file",
        action="store_true",
        default=False,
        help="Also save all reviews to a single combined CSV file",
    )

    return parser.parse_args()


def print_header():
    """Print script header with branding."""
    print("=" * 70)
    print(" " * 15 + "FINTECH APP REVIEW SCRAPER")
    print(" " * 10 + "Ethiopian Banking Apps - Google Play Store")
    print("=" * 70)
    print()


def print_summary(summary: Dict[str, Any], filepaths: Dict[str, Path]):
    """
    Print scraping summary to console.

    Args:
        summary (Dict[str, Any]): Summary statistics from scraping
        filepaths (Dict[str, Path]): Paths to saved CSV files
    """
    print("\n" + "=" * 70)
    print("SCRAPING SUMMARY")
    print("=" * 70)

    print(f"\nTotal Reviews Scraped: {summary['total_reviews']}")
    print(f"Banks Scraped: {summary['banks_scraped']}/3")
    print(f"Scraping Date: {summary['scraping_date']}")

    print("\nReviews per Bank:")
    for bank, count in summary["reviews_per_bank"].items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {bank}: {count} reviews")

    if filepaths:
        print("\nSaved Files:")
        for bank, filepath in filepaths.items():
            print(f"  • {bank}: {filepath}")

    # Data quality check
    print("\nData Quality Check:")
    target_per_bank = 400
    for bank, count in summary["reviews_per_bank"].items():
        if count >= target_per_bank:
            print(f"  ✓ {bank}: Target met ({count}/{target_per_bank})")
        elif count > 0:
            print(f"  ⚠ {bank}: Below target ({count}/{target_per_bank})")
        else:
            print(f"  ✗ {bank}: No reviews collected")

    print("\n" + "=" * 70)


def save_summary_json(summary: Dict[str, Any], filepath: Path):
    """
    Save scraping summary to JSON file.

    Args:
        summary (Dict[str, Any]): Summary statistics
        filepath (Path): Path to save JSON file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Summary saved to: {filepath}")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Print header
    print_header()

    # Configuration
    print("Configuration:")
    print(f"  Reviews per bank: {args.count}")
    print(f"  Output directory: {args.output_dir or Config.RAW_DATA_DIR}")
    print(f"  Separate files: {args.separate_files}")
    print(f"  Combined file: {args.combined_file}")
    print()

    # Initialize scraper
    print("Initializing scraper...")
    try:
        scraper = ReviewScraper()
        print("✓ Scraper initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize scraper: {e}")
        sys.exit(1)

    # Start scraping
    print("Starting scraping process...")
    print(f"Target: {args.count} reviews × 3 banks = {args.count * 3} total reviews")
    print()

    try:
        # Scrape all banks
        all_reviews = scraper.scrape_all_banks(count_per_bank=args.count)

        # Get summary
        summary = scraper.get_scraping_summary()

        # Save files
        output_dir = args.output_dir or Config.RAW_DATA_DIR
        filepaths = {}

        if args.separate_files:
            print("\nSaving separate CSV files...")
            filepaths = scraper.save_bank_reviews_separately(directory=output_dir)
            print(f"✓ Saved {len(filepaths)} bank files")

        if args.combined_file:
            print("\nSaving combined CSV file...")
            combined_filename = (
                f"all_banks_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            combined_path = scraper.save_to_csv(
                filename=combined_filename, directory=output_dir
            )
            print(f"✓ Saved combined file: {combined_path}")

        # Save summary JSON
        summary_file = args.summary_file or (output_dir / "scraping_summary.json")
        save_summary_json(summary, summary_file)

        # Print summary
        print_summary(summary, filepaths)

        # Success message
        if summary["total_reviews"] >= args.count * 3:
            print("\n🎉 SUCCESS! All targets met.")
        elif summary["total_reviews"] > 0:
            print(
                "\n⚠️  PARTIAL SUCCESS: Some reviews collected but targets not fully met."
            )
        else:
            print("\n❌ FAILED: No reviews collected.")
            sys.exit(1)

    except ReviewScraperError as e:
        print(f"\n✗ Scraping error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        print("Partial data may have been saved.")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
