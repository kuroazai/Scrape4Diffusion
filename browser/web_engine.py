import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.firefox.webdriver import FirefoxProfile
from selenium.webdriver.chrome import service
from selenium.webdriver.common import desired_capabilities


class Device:

    def __init__(self):
        executable_path = r"E:\Users\Kuro Azai\Python\Discord\Veda\discord\geckodriver.exe"
        profile = FirefoxProfile(r'C:\Users\Odin\AppData\Roaming\Mozilla\Firefox\Profiles\ub6qmkoz.default-release')
        options = Options()
        options.headless = False
        self.browser = webdriver.Firefox(profile, options=options, executable_path=executable_path)
        self.driver = self.browser

    def check_exists_by_xpath(self, xpath):
        try:
            element = self.driver.find_element(xpath)
            return element
        except NoSuchElementException:
            return None
        return element

    def click_xpath(self, xpath):
        element = self.check_exists_by_xpath(xpath)
        if element is not None:
            element.click()
            return True
        return False

    def go_back(self):
        self.driver.back()
        time.sleep(0.5)

    def load_page(self, url):
        self.driver.get(url)
        time.sleep(0.5)

    def scroll_to_bottom(self):
        SCROLL_PAUSE_TIME = 2.5
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        i = 2
        while i > 0:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            i -= 1

Browser = Device()