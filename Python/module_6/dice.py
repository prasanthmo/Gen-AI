from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv

# Setup ChromeDriver using webdriver-manager
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")  # Start maximized
options.add_argument("--incognito")         # Start in incognito mode
# options.add_argument("--headless")        # Uncomment this line to run in headless mode
driver = webdriver.Chrome(service=service, options=options)

# Function to scrape job data from Dice.com with pagination
def scrape_dice_jobs(base_url, pages_to_scrape=50):
    current_page = 1
    jobs_data = []

    # Open CSV file for writing the data
    with open('dice_systems_engineering_jobs.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Job Title', 'Location', 'Salary'])

        while current_page <= pages_to_scrape:
            url = f"{base_url}&page={current_page}"
            driver.get(url)
            time.sleep(5)  # Wait for the page to fully load

            try:
                # Wait for job cards to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.card"))
                )
                job_cards = driver.find_elements(By.CSS_SELECTOR, "div.card")

                if not job_cards:
                    print(f"No job cards found on page {current_page}. Ending scraping.")
                    break

                for job_card in job_cards:
                    # Extract job title
                    try:
                        job_title = job_card.find_element(By.XPATH, ".//a[contains(@class, 'card-title-link')]").text.strip()
                    except:
                        job_title = 'Not available'

                    # We only want jobs with "Systems Engineering" in the title
                    if "Systems Engineering" in job_title:
                        # Extract location
                        try:
                            location = job_card.find_element(By.XPATH, ".//span[contains(@class, 'job-location')]").text.strip()
                        except:
                            location = 'Location not available'

                        # Extract salary
                        try:
                            salary = job_card.find_element(By.XPATH, ".//span[contains(@class, 'posted-salary')]").text.strip()
                        except:
                            salary = 'Salary not listed'

                        # Save the job data in the list
                        jobs_data.append([job_title, location, salary])
                        writer.writerow([job_title, location, salary])

                print(f"Scraped page {current_page}.")
                current_page += 1
            except Exception as e:
                print(f"Error occurred on page {current_page}: {e}. Ending scraping.")
                break

    print(f"Scraping completed. Data saved to 'dice_systems_engineering_jobs.csv'.")

# Base URL of the Dice job search page (for Systems Engineering)
base_url = 'https://www.dice.com/jobs?q=systems%20engineering&countryCode=US&radius=30&radiusUnit=mi&language=en'

# Scrape jobs with pagination for 50 pages
scrape_dice_jobs(base_url, pages_to_scrape=50)

# Close the browser when done
driver.quit()
