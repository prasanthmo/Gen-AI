import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the Chrome WebDriver
service = Service('C:\\Users\\mohan\\GenAi\\chromedriver.exe')
driver = webdriver.Chrome(service=service)

# Define the URL and application ID range
url = "https://ceoaperolls.ap.gov.in/AP_MLC_2024/ERO/Status_Update_2024/knowYourApplicationStatus.aspx"
app_id_range = range(1001, 1011)  # F18-0001001 to F18-0001010

# Open the URL
driver.get(url)

# Wait for the page to load and select the Graduate category
try:
    graduate_link = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.LINK_TEXT, "Graduate"))
    )
    graduate_link.click()
except Exception as e:
    print(f"Error selecting Graduate category: {e}")
    driver.quit()

# Click the radio button for the application ID
try:
    radio_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "GraduateAppID"))
    )
    radio_button.click()
except Exception as e:
    print(f"Error clicking radio button: {e}")
    driver.quit()

# Initialize a list to store scraped data
scraped_data = []

# Loop through the application ID range
for i in app_id_range:
    app_id = f"F18-000{i}"
    print(f"Scraping {app_id}...")

    # Find the input field for the application ID using its ID
    try:
        app_id_input = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "TextBox2"))  # Change to the correct input ID
        )
        app_id_input.clear()  # Clear any existing text
        app_id_input.send_keys(app_id)

        # Click the search button
        search_button = driver.find_element(By.ID, "btnGraduates")
        search_button.click()

        # Wait for the results to load
        time.sleep(2)  # Adjust the sleep time as necessary

        # Scrape the data from the table
        try:
            # Wait for the table to be present
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "GridViewGraduate"))
            )

            # Extract the rows from the table
            rows = table.find_elements(By.TAG_NAME, "tr")

            # Extract the headers
            headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, "th")]

            # Extract the data from the second row (index 1)
            data = [cell.text for cell in rows[1].find_elements(By.TAG_NAME, "td")]
            if data:  # Check if there's any data in the row
                scraped_data.append([app_id] + data)
        except Exception as e:
            print(f"Error scraping data for {app_id}: {e}")

    except Exception as e:
        print(f"Error processing {app_id}: {e}")

# Save the scraped data to a CSV file
try:
    with open('mlc_application_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['App ID'] + headers)  # Write headers
        writer.writerows(scraped_data)  # Write data rows

    print("Scraping complete. Data saved to 'mlc_application_data.csv'.")
except Exception as e:
    print(f"Error saving data to CSV: {e}")

# Close the browser
driver.quit()
