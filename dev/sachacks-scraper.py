from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# List of URLs to scrape
urls = [
    "https://sachacks.io/agenda"
]

# CSS Selector for all clickable divs
clickable_div_selector = "div.cursor-pointer"

def scrape_page_with_selenium(url):
    """Scrape a webpage using Selenium by clicking all interactive divs."""
    
    driver = webdriver.Chrome()
    driver.get(url)
    driver.implicitly_wait(5)  # Wait for elements to load

    print(f"Scraping: {url}")

    try:
        # Wait for clickable divs to appear
        clickable_divs = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, clickable_div_selector))
        )

        count = len(clickable_divs)
        print(f"Found {count} clickable divs. Clicking each...")

        for div in clickable_divs:
            try:
                # Scroll element into view
                driver.execute_script("arguments[0].scrollIntoView();", div)
                time.sleep(0.5)  # Ensure it is scrolled into view
                
                # Click the button
                div.click()
                time.sleep(1)  # Wait 1 second for content to load
            except Exception as e:
                print(f"Skipping div due to error: {e}")

        # Extract all text after interactions
        page_content = driver.find_element(By.TAG_NAME, "body").text

    except Exception as e:
        print(f"Error scraping {url}: {e}")  # Print full error message
        page_content = ""

    finally:
        driver.quit()  # Close the browser

    return page_content


# Scrape each URL
scraped_data = {url: scrape_page_with_selenium(url) for url in urls}

# Save results to CSV
df = pd.DataFrame(scraped_data.items(), columns=["URL", "Content"])
df.to_csv("sachacks_scraped_data_selenium_fixed.csv", index=False)

print("Scraping completed. Data saved to sachacks_scraped_data_selenium_fixed.csv")
