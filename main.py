import os
import csv
import glob

from dotenv import load_dotenv

from scripts.phase2_sentiment_analysis_classic_ml import SentimentAnalysisPreprocessor
from scripts.phase1_scrapper import RedditScraper


def concatenate_tsv_files(list_of_files: list, destination_file: str):
    collected_posts = [
        post for individual_file in list_of_files for post in
        csv.DictReader(open(individual_file, 'r', newline='', encoding='utf-8'), delimiter='\t')
    ]

    with open(destination_file, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=collected_posts[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(collected_posts)


if __name__ == '__main__':
    load_dotenv()

    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    user_agent_id = os.getenv('USER_AGENT')

    subreddit_selection = [
        'ukraine', 'ukrainianconflict', 'russia', 'europe', 'military', 'geopolitics',
        'worldnews', 'worldpolitics', 'foreignpolicy', 'war', 'internationalpolitics',
        'globaldevelopment', 'humanrights', 'cybersecurity', 'energy', 'globalhealth',
        'science', 'environment', 'technology'
    ]

    scraper_instance = RedditScraper(client_id, client_secret, user_agent_id, subreddit_selection, num_posts=10000)
    # scraper_instance.scrape(output_file="csv_files/data/reddit_posts_20230503.tsv")
    # Uncomment the above line if .tsv files are not present

    files_with_reddit_data = glob.glob("csv_files/data/*.tsv")
    concatenate_tsv_files(files_with_reddit_data, "csv_files/data/post_from_reddit_main.tsv")

    preprocessor_instance = SentimentAnalysisPreprocessor(
        input_file='csv_files/data/post_from_reddit_main.tsv',
        output_file='csv_files/processed_data/preprocessed_data.tsv'
    )
    preprocessor_instance.preprocess()
