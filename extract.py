from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import os
from dotenv import load_dotenv

def fetch_rendered_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()
        browser.close()
        return content

def extract_table_from_html(html_content):
    # Parse the HTML content with Beautiful Soup
    soup = BeautifulSoup(html_content, 'lxml')
    # Extract the table
    table = soup.find("table")
    # Extract table rows
    rows = table.find_all('tr')
    # Parse the rows into a list of lists
    data = []
    for row in rows:
        cols = row.find_all(['td', 'th'])
        cols = [ele.text.strip().replace('\n', '') for ele in cols]
        data.append(cols)
    # Convert to DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    # Drop the first three columns and "Future Value"
    df = df.drop(columns=df.columns[[0, 1, 2, -1]])
    # Rename the "Team" column to "TEA"
    df = df.rename(columns={"Team": "TEA"})
    return df


def save_to_csv(df):
    # Create a directory named "data" if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # Extract the first week column number from the DataFrame columns
    first_week_column = df.columns[1]

    # Construct the filename
    filename = f"./data/2023nflweek{first_week_column}.csv"

    # Save the DataFrame to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Main execution

def extract_data():
    url = os.getenv('SCRAPE_URL')
    print("\rFetching content...")
    html_content = fetch_rendered_html(url)
    print("\rExtracting data...")
    df = extract_table_from_html(html_content)
    save_to_csv(df)
    return df

def main():
    extract_data()

if __name__ == '__main__':
    main()
