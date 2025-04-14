!pip install requests beautifulsoup4 pandas

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Target base URL
base_url = 'http://quotes.toscrape.com/page/{}/'
all_quotes = []

# Loop through multiple pages
for page in range(1, 6):  # Adjust the range based on how many pages you want to scrape
    url = base_url.format(page)
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve page {page}")
        break

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all quote containers
    quote_blocks = soup.find_all('div', class_='quote')

    if not quote_blocks:
        print("No more quotes found. Ending scrape.")
        break

    # Extract data from each quote block
    for quote in quote_blocks:
        text = quote.find('span', class_='text').get_text(strip=True)
        author = quote.find('small', class_='author').get_text(strip=True)
        tags = [tag.get_text(strip=True) for tag in quote.find_all('a', class_='tag')]
        
        all_quotes.append({
            'quote': text,
            'author': author,
            'tags': ", ".join(tags)
        })

# Convert to DataFrame
df = pd.DataFrame(all_quotes)

# Save to CSV
df.to_csv('scraped_quotes.csv', index=False, encoding='utf-8')

df
