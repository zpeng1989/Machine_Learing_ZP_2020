from bs4 import BeautifulSoup

html = """
        <div class = 'full_name'><span style='font-weight:blod'>
        Masegep Azra"
"""

soup = BeautifulSoup(html, 'lxml')

print(soup)