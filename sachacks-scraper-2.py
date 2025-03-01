from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# List of URLs to scrape
urls = [
    "https://sachacks.io/",
    "https://sachacks.io/tracks",
    "https://sachacks.io/agenda"
]

def scrape_entire_page(url):
    """Scrape an entire webpage using Selenium without interacting with elements."""
    
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    driver.implicitly_wait(10)  # Wait for initial page load

    print(f"Scraping: {url}")

    try:
        # Scroll down to trigger JavaScript lazy-loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Wait for additional content to load

        # Extract all text from the page
        page_content = driver.find_element(By.TAG_NAME, "body").text

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        page_content = ""

    finally:
        driver.quit()  # Close the browser

    return page_content

# Scrape each URL
scraped_data = {url: scrape_entire_page(url) for url in urls}

# Save results to CSV
df = pd.DataFrame(scraped_data.items(), columns=["URL", "Content"])
df.to_csv("sachacks_scraped_data_full.csv", index=False)

print("Scraping completed. Data saved to sachacks_scraped_data_full.csv")
