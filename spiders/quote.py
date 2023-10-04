import scrapy
import pandas as pd
from scrapy.crawler import CrawlerProcess
import time

class QuoteSpider(scrapy.Spider):
    name = "quote"
    allowed_domains = ["www.webmd.com"]
    start_urls = ["https://reviews.webmd.com/vitamins-supplements/ingredientreview-794-5-htp"]
    review_list = []

    def parse(self, response, **kwargs):
        review_details = response.css('div.review-details')
        dates = review_details.css('div.date::text').getall()
        details = review_details.css('div.details::text').getall()
        scores = review_details.css('strong::text').getall()
        reviews = review_details.css('p.description-text::text').getall()

        for date, detail, score, review in zip(dates, details, scores, reviews):
            self.review_list.append({"date": date, "detail": detail, 'score': score, "review": review})

        next_page_link = response.css('a.page-link::attr(href)').get()
        if next_page_link:
            next_page_url = response.urljoin(next_page_link)
            yield scrapy.Request(next_page_url, callback=self.parse)

    def close(self, reason):
        # Создание DataFrame с данными отзывов
        df = pd.DataFrame(self.review_list)

        # Сохранение DataFrame в Excel-файл
        df.to_excel('C:/Users/lyuob/Documents/quotes.xlsx', index=False)


# Создание экземпляра процесса скрапинга
process = CrawlerProcess()
process.crawl(QuoteSpider)
process.start()
