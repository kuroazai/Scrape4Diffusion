import time

import requests
from bs4 import BeautifulSoup
from browser import web_engine
import config as cfg
import os


class Scrap4Diffusion:

    def __init__(self, cfg=cfg, name="", profile=None):
        self.name = name
        self.insta_url = "https://www.instagram.com"
        self.profile_page = profile
        self.out_dir = os.path.join(os.getcwd(), f"output/{name}")
        toggle = self.check_output_folder()
        self.browser = web_engine.Browser
        if toggle:
            print(f"Scraping {name}...")
            self.run_instance()
        else:
            print(f"data for {name} already exists.")


    def check_output_folder(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            return True
        else:
            return False

    def get_browser_content(self):
        return self.browser.driver.page_source

    def run_instance(self):
        self.browser.load_page(self.profile_page)
        self.browser.scroll_to_bottom()
        self.soup = self.load_soup(self.get_browser_content())
        divs = self.find_div_class('_aagv')
        for div in divs:
            soup = BeautifulSoup(str(div), "html.parser")
            img_tag = soup.find('img')
            src = img_tag['src']
            self.save_image(src)


    def process_images(self, image_urls: list):
        for x in image_urls:
            pass

    def save_image(self, image_url: str):
        count = len(os.listdir(self.out_dir)) + 1
        #name = f"insta_{image_url.split('/')[5].split('?')[0]}"
        name = f"{self.name} ({count}).jpg"
        if os.path.exists(os.path.join(self.out_dir, name)):
            return
        response = requests.get(image_url)

        path = os.path.join(self.out_dir, name)
        with open(path, "wb") as file:
            file.write(response.content)

    @staticmethod
    def load_soup(content: str):
        return BeautifulSoup(content, "html.parser")

    def find_div_class(self, class_name: str):
        return self.soup.find_all("div", {"class": class_name})

    def find_a_href(self, div):
        return div.find("a").get("href")

    def find_all_a_href(self, divs):
        hrefs = []
        for div in divs:
            href = self.find_a_href(div)
            hrefs.append(href)
        return hrefs


sources = {'name': 'instgram url handle'}


for source in sources:
    print(source, source[source])
    scraper = Scrap4Diffusion(name=source, profile=sources[source])
