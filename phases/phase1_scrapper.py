import re
import praw
import pandas as pd
import logging


class RedditDataScraper:
    def __init__(self, app_id, app_secret, app_user_agent, target_subreddits, posts_limit=100):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        for logger_id in ("praw", "prawcore"):
            current_logger = logging.getLogger(logger_id)
            current_logger.setLevel(logging.DEBUG)
            current_logger.addHandler(handler)

        self.reddit_instance = praw.Reddit(client_id=app_id, client_secret=app_secret, user_agent=app_user_agent)
        self.target_subreddits = target_subreddits
        self.posts_limit = posts_limit

    def collect_data(self, output_filename):
        collected_posts = []
        count_1, count_2 = 0, 0

        for subreddit in self.target_subreddits:
            count_1 += 1
            for submission in self.reddit_instance.subreddit('popular').top(limit=self.posts_limit, time_filter='all'):
                if submission.is_self:
                    collected_posts.append({
                        'author': submission.author.name if submission.author else '',
                        'title': submission.title,
                        'score': submission.score,
                        'id': submission.id,
                        'url': submission.url,
                        'num_comments': submission.num_comments,
                        'created': pd.to_datetime(submission.created_utc, unit='s'),
                        'subreddit': subreddit,
                        'selftext': re.sub(r'["\n\t]', ' ', submission.selftext)
                    })
                    count_2 += 1
                    print(f"INFO: Processed count_1: {count_1} and count_2: {count_2}")

        data_frame = pd.DataFrame(collected_posts)
        data_frame.to_csv(output_filename, index=False, sep='\t')
