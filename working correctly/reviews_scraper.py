import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse, parse_qs


class ReviewSpider(scrapy.Spider):
    name = 'reviewspider'
    start_url = 'https://reviews.webmd.com/vitamins-supplements/ingredientreview-794-5-htp?conditionid=&sortval=1&page=1&next_page=true'

    def start_requests(self):
        yield scrapy.Request(url=self.start_url, callback=self.parse)

    custom_settings = {
        'FEEDS': {'reviews.csv': {'format': 'csv'}}
    }

    def parse(self, response):
        for review_details in response.css('div.review-details-holder'):
            reviews = review_details.css('p.description-text::text').getall()

            if not reviews:  # Если обычные отзывы не найдены
                reviews_1 = review_details.css('span.showSec::text').getall()
                reviews_2 = review_details.css('span.hiddenSec::text').getall()
                reviews = reviews_1 + reviews_2  # Совмещаем содержимое обоих столбцов отзывов

            yield {
                "dates": review_details.css('div.date::text').getall(),
                "details": review_details.css('div.details::text').getall(),
                "scores": review_details.css('strong::text').getall(),
                "reviews": reviews,
                # Здесь уже будут или "обычные" отзывы, или объединённые отзывы из двух других столбцов
            }

        parsed_url = urlparse(response.url)
        current_page = int(parse_qs(parsed_url.query)['page'][0])
        next_page = current_page + 1
        if next_page <= 20:
            next_url = f'https://reviews.webmd.com/vitamins-supplements/ingredientreview-794-5-htp?conditionid=&sortval=1&page={next_page}&next_page=true'
            yield response.follow(next_url, callback=self.parse)


process = CrawlerProcess()
process.crawl(ReviewSpider)
process.start()