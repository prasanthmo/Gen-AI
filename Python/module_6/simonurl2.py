import requests
from bs4 import BeautifulSoup
import re
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Function to create a PDF file from text
def create_pdf(file_path, content):
    pdf = canvas.Canvas(file_path, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    # Split content into lines and write to PDF
    for i, line in enumerate(content.split('\n')):
        pdf.drawString(100, 750 - i * 15, line)  # Adjust Y position for each line

    pdf.save()

# Function to get all URLs from the website
def get_internal_urls(url):
    internal_urls = []
    external_urls = []
    base_url = re.match(r'https?://([^/]+)', url).group(0)

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to access {url}")
        return internal_urls, external_urls

    soup = BeautifulSoup(response.content, 'html.parser')

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.startswith('/'):  # Relative URLs
            internal_urls.append(base_url + href)
        elif base_url in href:  # Absolute internal URLs
            internal_urls.append(href)
        else:  # External URLs
            external_urls.append(href)

    return internal_urls, external_urls

# Function to scrape content from a URL
def scrape_content(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract relevant content; modify this according to the website's structure
        title = soup.title.string if soup.title else 'No Title'
        paragraphs = soup.find_all('p')
        content = [title] + [p.get_text() for p in paragraphs]

        return "\n".join(content)

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def main():
    base_url = 'https://www.simonfurniture.com/'  # The main website URL
    internal_urls, external_urls = get_internal_urls(base_url)

    # Save external URLs to a text file
    with open('external_links.txt', 'w') as ext_file:
        for ext_url in external_urls:
            ext_file.write(ext_url + '\n')

    # Create a directory for PDF files
    if not os.path.exists('pdfs'):
        os.makedirs('pdfs')

    # Scrape content from internal URLs and create PDFs
    for url in internal_urls:
        content = scrape_content(url)
        if content:
            pdf_file_path = os.path.join('pdfs', f"{re.sub(r'[^a-zA-Z0-9]', '_', url[8:])}.pdf")
            create_pdf(pdf_file_path, content)
            print(f"Created PDF: {pdf_file_path}")

if __name__ == "__main__":
    main()
