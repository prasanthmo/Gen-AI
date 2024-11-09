import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up the Chrome WebDriver using WebDriverManager
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Function to modify CSS styles dynamically
def modify_css():
    # Change the background color of the body
    driver.execute_script("""
        document.body.style.backgroundColor = '#f0f0f0';
    """)

    # Change the font color and size of all <p> elements
    driver.execute_script("""
        var paragraphs = document.getElementsByTagName('p');
        for (var i = 0; i < paragraphs.length; i++) {
            paragraphs[i].style.color = 'blue';
            paragraphs[i].style.fontSize = '18px';
        }
    """)

    # Add a red border to all buttons
    driver.execute_script("""
        var buttons = document.getElementsByTagName('button');
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].style.border = '3px solid red';
        }
    """)

# Function to scrape data from a specific page
def scrape_data():
    # Open the page you want to scrape
    driver.get('https://example.com')  # Replace with the target URL

    # Wait for the page to load and the necessary element to appear
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, 'TextBox2'))  # Adjust according to the element on the page
    )

    # Example input IDs loop from F18-0000001 to F18-0064000
    app_ids = [f'F18-{str(i).zfill(7)}' for i in range(1, 64001)]

    data = []  # To store the scraped data

    for app_id in app_ids:
        try:
            # Input the application ID in the form
            input_element = driver.find_element(By.ID, 'TextBox2')
            input_element.clear()
            input_element.send_keys(app_id)

            # Submit the form if needed (you might need to adjust this part based on your form)
            submit_button = driver.find_element(By.ID, 'submit_button_id')  # Replace with actual submit button ID
            submit_button.click()

            # Wait for the results to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, 'results_table_id'))  # Adjust for the actual results table
            )

            # Extract data from the table
            table_rows = driver.find_elements(By.CSS_SELECTOR, '#results_table_id tr')
            for row in table_rows:
                columns = row.find_elements(By.TAG_NAME, 'td')
                row_data = [col.text for col in columns]
                data.append(row_data)

            # Modify CSS after each iteration
            modify_css()

        except Exception as e:
            print(f"Failed for App ID {app_id}: {e}")

    # Save the scraped data to a CSV file
    columns = ['App ID', 'PS_NO', 'SLNO', 'Name', 'Relation Name', 'House Number', 'Age', 'AERO Comments', 'Ero Comments', 'App Status']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('mlc_application_data.csv', index=False)

    print("Data scraping complete. Saved to 'mlc_application_data.csv'.")

# Main execution
scrape_data()

# Close the browser
driver.quit()

