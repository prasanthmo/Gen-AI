import requests
from bs4 import BeautifulSoup

def scrape_sublinks_to_file(url, output_filename):
    # Send a request to fetch the page content
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all anchor tags with 'href' attributes
        links = soup.find_all('a', href=True)

        # Save sublinks to a file
        with open(output_filename, 'w', encoding='utf-8') as file:
            for link in links:
                href = link['href']
                # Check if the link is relative and append the base URL
                if href.startswith('/'):
                    href = url + href
                elif not href.startswith('http'):
                    continue  # Skip non-valid URLs
                file.write(href + '\n')

        print(f"Sublinks scraped and saved to '{output_filename}' successfully.")
    else:
        print("Failed to retrieve the page.")

# Example usage
url = 'https://www.simonfurniture.com/'  # Replace with your URL
output_filename = 'sublinks.txt'  # Replace with your desired output file name

scrape_sublinks_to_file(url, output_filename)
