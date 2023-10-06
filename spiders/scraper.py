import scrapy

class ReviewSpider(scrapy.Spider):
    name = 'reviewspider'
    start_urls = ['https://reviews.webmd.com/vitamins-supplements/ingredientreview-794-5-htp?conditionid=&sortval=1&page=1&next_page=true']

    custom_settings = {
        'FEEDS': { 'quotes.csv': { 'format': 'csv', }}
        }
    
    def parse(self, response):

        for review_details in response.css('div.review-details'):
            yield {
                "dates": review_details.css('div.date::text').getall(),
                "details": review_details.css('div.details::text').getall(),
                "scores": review_details.css('strong::text').getall(),
                "reviews": review_details.css('p.description-text::text').getall()
            }

        num_pages = 20
        
        for page in range(2, num_pages + 1):
            next_page = f'https://reviews.webmd.com/vitamins-supplements/ingredientreview-794-5-htp?conditionid=&sortval=1&page={page}&next_page=true'
            print(next_page) # bag
            yield response.follow(next_page, callback=self.parse)