import requests
from bs4 import BeautifulSoup

def scrape_text_to_file(url, output_filename):
    # Send a request to fetch the page content
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract all text from the page
        page_text = soup.get_text(separator='\n', strip=True)

        # Save the extracted text to a file
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(page_text)

        print(f"Text data scraped and saved to '{output_filename}' successfully.")
    else:
        print("Failed to retrieve the page.")

# Example usage
url = 'https://www.simonfurniture.com/dining-room'  # Replace with your URL
output_filename = 'scraped_data.txt'  # Replace with your desired output file name

scrape_text_to_file(url, output_filename)
