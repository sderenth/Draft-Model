import scrapy
import pandas as pd
from bs4 import BeautifulSoup

df = pd.DataFrame(pd.read_csv('2018playerInfo.csv'))
df['href'] = 'https://basketball.realgm.com' + df['href'].astype(str)
player_hrefs = df['href'].tolist()

class preNBAdata(scrapy.Spider):
    name = "playerStats"
    start_urls = player_hrefs

    def parse(self, response):
        soup = BeautifulSoup(response.body, 'html.parser')

        tot_tab = soup.find_all('h2', text='NCAA Season Stats - Totals')[0].next_sibling.find_all('tr')[1:-1]
        adv_tab = soup.find_all('h2', text='NCAA Season Stats - Advanced Stats')[0].next_sibling.find_all('tr')[1:-1]

        for i in range(len(tot_tab)):
            tot_row_data = []
            adv_row_data = []
            for td in tot_tab[i].find_all('td'):
                tot_row_data.append(td.getText())

            for td in adv_tab[i].find_all('td'):
                adv_row_data.append(td.getText())

            columns = ['href', 'season', 'team', 'class', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%',
                       'FTM', 'FTA', 'FT%', 'OFF', 'DEF', 'TRB', 'AST', 'STL', 'BLK', 'PF', 'TOV', 'PTS', 'TS%', 'eFG%',
                       'ORB%', 'DRB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%', 'Total S%', 'PPR', 'PPS', 'ORtg',
                       'DRtg', 'PER']

            season_data = {
                'href': response.url,
                'season': tot_row_data[0],
                'team' : tot_row_data[1],
                'class': tot_row_data[2],
                'GP': tot_row_data[3],
                'GS': tot_row_data[4],
                'MIN': tot_row_data[5],
                'FGM': tot_row_data[6],
                'FGA': tot_row_data[7],
                'FG%': tot_row_data[8],
                '3PM': tot_row_data[9],
                '3PA': tot_row_data[10],
                '3P%': tot_row_data[11],
                'FTM': tot_row_data[12],
                'FTA': tot_row_data[13],
                'FT%': tot_row_data[14],
                'OFF': tot_row_data[15],
                'DEF': tot_row_data[16],
                'TRB': tot_row_data[17],
                'AST': tot_row_data[18],
                'STL': tot_row_data[19],
                'BLK': tot_row_data[20],
                'PF': tot_row_data[21],
                'TOV': tot_row_data[22],
                'PTS': tot_row_data[23],
                'TS%': adv_row_data[5],
                'eFG%': adv_row_data[6],
                'ORB%': adv_row_data[7],
                'DRB%': adv_row_data[8],
                'TRB%': adv_row_data[9],
                'AST%': adv_row_data[10],
                'TOV%': adv_row_data[11],
                'STL%': adv_row_data[12],
                'BLK%': adv_row_data[13],
                'USG%': adv_row_data[14],
                'Total S%': adv_row_data[15],
                'PPR': adv_row_data[16],
                'PPS': adv_row_data[17],
                'ORtg': adv_row_data[18],
                'DRtg': adv_row_data[19],
                'PER': adv_row_data[20]}

            df = pd.DataFrame(season_data, index=[0], columns=columns)
            df.to_csv('PlayerStats.csv', mode='a', header=False)