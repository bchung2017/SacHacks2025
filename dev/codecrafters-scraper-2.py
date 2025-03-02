import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")

# Target URL (Codecrafters PHP track)
url = "https://app.codecrafters.io/tracks/php"

# Send a GET request to fetch the page content
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all course stage links and their corresponding text
    data = []
    stop_words = set(stopwords.words("english"))

    # Find all course sections
    for section in soup.find_all("div", class_="bg-white dark:bg-gray-850 rounded-md shadow-sm border border-gray-200 dark:border-white/5 relative w-full group mb-4"):
        # Extract section title
        section_title_div = section.find("div", class_="text-xl font-semibold text-gray-800 dark:text-gray-200")
        section_title = section_title_div.get_text(strip=True) if section_title_div else "Unknown Section"

        # Extract course links within this section
        for link in section.find_all("a", class_="ember-view block hover:bg-gray-50 dark:hover:bg-gray-700/50 py-1.5 -mx-1.5 px-1.5 rounded"):
            href = link.get("href")
            text_div = link.find("div", class_="prose dark:prose-invert prose-sm")

            if href and text_div:
                full_link = f"https://app.codecrafters.io{href}"  # Ensure full URL
                text = text_div.get_text(strip=True)

                # Combine section title and text for better tagging
                combined_text = f"{section_title} {text}"

                # Generate tags from the combined text (remove stopwords)
                words = set(re.findall(r'\b\w+\b', combined_text.lower())) - stop_words
                tags = " | ".join(words)

                data.append((full_link, text, section_title, tags))

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Full Link", "Text", "Section", "Tags"])

    # Save to CSV
    output_file = "extracted_codecrafters_links_with_combined_tags.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    # Display extracted data
    print(df)

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
