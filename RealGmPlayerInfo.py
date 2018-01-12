import scrapy
from bs4 import BeautifulSoup

class preNBAdata(scrapy.Spider):
    name = "playerInfo"
    start_urls = ['https://basketball.realgm.com/ncaa/players/2018/A',
                  'https://basketball.realgm.com/ncaa/players/2018/B',
                  'https://basketball.realgm.com/ncaa/players/2018/C',
                  'https://basketball.realgm.com/ncaa/players/2018/D',
                  'https://basketball.realgm.com/ncaa/players/2018/E',
                  'https://basketball.realgm.com/ncaa/players/2018/F',
                  'https://basketball.realgm.com/ncaa/players/2018/G',
                  'https://basketball.realgm.com/ncaa/players/2018/H',
                  'https://basketball.realgm.com/ncaa/players/2018/I',
                  'https://basketball.realgm.com/ncaa/players/2018/J',
                  'https://basketball.realgm.com/ncaa/players/2018/K',
                  'https://basketball.realgm.com/ncaa/players/2018/L',
                  'https://basketball.realgm.com/ncaa/players/2018/M',
                  'https://basketball.realgm.com/ncaa/players/2018/N',
                  'https://basketball.realgm.com/ncaa/players/2018/O',
                  'https://basketball.realgm.com/ncaa/players/2018/P',
                  'https://basketball.realgm.com/ncaa/players/2018/Q',
                  'https://basketball.realgm.com/ncaa/players/2018/R',
                  'https://basketball.realgm.com/ncaa/players/2018/S',
                  'https://basketball.realgm.com/ncaa/players/2018/T',
                  'https://basketball.realgm.com/ncaa/players/2018/U',
                  'https://basketball.realgm.com/ncaa/players/2018/V',
                  'https://basketball.realgm.com/ncaa/players/2018/W',
                  'https://basketball.realgm.com/ncaa/players/2018/X',
                  'https://basketball.realgm.com/ncaa/players/2018/Y',
                  'https://basketball.realgm.com/ncaa/players/2018/Z']

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')

        tab_rows = soup.find_all('tr')[1:]

        for row in tab_rows:
            tds = row.find_all('td')
            yield {
                'player_name': tds[0].getText(),
                'href': tds[0].a['href'],
                'pos': tds[1].getText(),
                'height': tds[2].getText(),
                'weight': tds[3].getText(),
                'team': tds[4].getText(),
                'birthday_code': tds[6]['rel']
            }