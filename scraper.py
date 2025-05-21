import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import time
import random
from urllib.parse import urljoin

# Create necessary directories
os.makedirs("dataset", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
}

# Function to safely make requests with retries
def safe_request(url, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                sleep_time = delay * (attempt + 1) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to retrieve {url} after {max_retries} attempts")
                return None

# Function to scrape articles from Prothom Alo
def scrape_prothom_alo():
    print("Scraping Prothom Alo...")
    base_url = "https://www.prothomalo.com"
    search_url = f"{base_url}/search?q=শেখ মুজিবুর রহমান"
    
    articles = []
    page = 1
    max_pages = 5  # Limit to prevent excessive scraping
    
    while page <= max_pages:
        print(f"Scraping page {page}...")
        current_url = f"{search_url}&page={page}" if page > 1 else search_url
        
        response = safe_request(current_url)
        if not response:
            break
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article elements - this selector might need adjustment based on the actual site structure
        article_elements = soup.select("div.article-item, div.news-card")
        
        if not article_elements:
            print("No more articles found.")
            break
            
        for article in article_elements:
            try:
                title_elem = article.select_one("h2, .title")
                link_elem = article.select_one("a")
                summary_elem = article.select_one(".summary, .excerpt, p")
                
                if title_elem and link_elem:
                    title = title_elem.text.strip()
                    link = urljoin(base_url, link_elem.get('href', ''))
                    summary = summary_elem.text.strip() if summary_elem else "No summary available"
                    
                    # Get the full article content
                    article_data = get_article_content(link)
                    
                    if article_data and "Sheikh Mujib" in article_data.get("content", "") or "শেখ মুজিব" in article_data.get("content", ""):
                        articles.append({
                            "title": title,
                            "source": "Prothom Alo",
                            "url": link,
                            "summary": summary,
                            "content": article_data.get("content", ""),
                            "published_date": article_data.get("date", "Unknown")
                        })
                        print(f"Collected article: {title}")
            except Exception as e:
                print(f"Error processing article: {e}")
                
        page += 1
        # Polite delay
        time.sleep(random.uniform(2, 5))
    
    print(f"Collected {len(articles)} articles from Prothom Alo")
    return articles

# Function to get full article content
def get_article_content(url):
    response = safe_request(url)
    if not response:
        return None
        
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract article content - selectors need adjustment for the specific site
    content_elem = soup.select_one("div.article-content, div.story-content, div.news-content")
    date_elem = soup.select_one("time, .date, .publication-date")
    
    if content_elem:
        # Remove unwanted elements like ads, related articles, etc.
        for unwanted in content_elem.select(".advertisement, .related-news, .promotion"):
            unwanted.decompose()
            
        content = content_elem.get_text(separator="\n").strip()
        date = date_elem.text.strip() if date_elem else "Unknown"
        
        return {
            "content": content,
            "date": date
        }
    
    return None

# Function to scrape Wikipedia
def scrape_wikipedia():
    print("Scraping Bangla Wikipedia...")
    # URLs for Sheikh Mujibur Rahman in Bangla Wikipedia
    wiki_url = "https://bn.wikipedia.org/wiki/শেখ_মুজিবুর_রহমান"
    
    response = safe_request(wiki_url)
    if not response:
        return []
        
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get the main content
    content_div = soup.select_one("#mw-content-text")
    if not content_div:
        print("Failed to find content on Wikipedia page")
        return []
        
    # Extract title
    title = soup.select_one("#firstHeading").text.strip() if soup.select_one("#firstHeading") else "শেখ মুজিবুর রহমান"
    
    # Extract sections
    sections = []
    current_section = {"title": "Introduction", "content": ""}
    
    # Process all paragraphs, headings and lists in the content
    for element in content_div.select("p, h2, h3, h4, h5, ul, ol"):
        if element.name in ['h2', 'h3', 'h4', 'h5']:
            # Save the previous section and start a new one
            if current_section["content"].strip():
                sections.append(current_section)
            
            # Create new section
            section_title = element.get_text().strip()
            # Remove edit links from section title
            section_title = section_title.split('[')[0].strip()
            current_section = {"title": section_title, "content": ""}
        else:
            # Add to current section content
            current_section["content"] += element.get_text() + "\n\n"
    
    # Add the last section
    if current_section["content"].strip():
        sections.append(current_section)
    
    # Extract images
    images = []
    for img in content_div.select('a.image img'):
        try:
            image_url = img.get('src', '')
            if image_url and not image_url.startswith('http'):
                image_url = 'https:' + image_url
            
            if image_url:
                alt_text = img.get('alt', 'Sheikh Mujibur Rahman image')
                images.append({
                    "url": image_url,
                    "alt_text": alt_text
                })
        except Exception as e:
            print(f"Error processing image: {e}")
    
    # Extract references
    references = []
    refs_div = soup.select_one('.references')
    if refs_div:
        for li in refs_div.select('li'):
            ref_text = li.get_text().strip()
            if ref_text:
                references.append(ref_text)
    
    wiki_data = [{
        "title": title,
        "source": "Bangla Wikipedia",
        "url": wiki_url,
        "sections": sections,
        "images": images,
        "references": references
    }]
    
    print("Successfully scraped Bangla Wikipedia")
    return wiki_data

# Main function to run both scrapers and save results
def main():
    all_data = {
        "news_articles": [],
        "wikipedia": []
    }
    
    # Scrape from Prothom Alo
    prothom_alo_articles = scrape_prothom_alo()
    all_data["news_articles"].extend(prothom_alo_articles)
    
    # Scrape from Wikipedia
    wiki_data = scrape_wikipedia()
    all_data["wikipedia"] = wiki_data
    
    # Save the scraped data to JSON files
    print("Saving data to JSON files...")
    
    with open("dataset/mujib71_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    # Convert news articles to CSV for easier usage
    if all_data["news_articles"]:
        df = pd.DataFrame(all_data["news_articles"])
        df.to_csv("dataset/news_articles.csv", index=False, encoding="utf-8")
    
    # Create a metadata file
    metadata = {
        "dataset_name": "Sheikh Mujibur Rahman Bangla NLP Dataset",
        "version": "1.0.0",
        "description": "A dataset containing Bangla text about Sheikh Mujibur Rahman collected from Prothom Alo and Bangla Wikipedia.",
        "sources": ["Prothom Alo", "Bangla Wikipedia"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "article_count": len(all_data["news_articles"]),
        "wikipedia_page_count": len(all_data["wikipedia"])
    }
    
    with open("dataset/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print("Dataset creation complete!")
    print(f"Total news articles: {len(all_data['news_articles'])}")
    print(f"Wikipedia pages: {len(all_data['wikipedia'])}")

if __name__ == "__main__":
    main()
