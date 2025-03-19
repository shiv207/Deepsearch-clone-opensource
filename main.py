from groq import Groq
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import nltk
from duckduckgo_search import DDGS
import time
import random
from urllib.parse import urlparse, quote_plus
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from googlesearch import search
import concurrent.futures
import wikipediaapi
from tenacity import retry, stop_after_attempt, wait_exponential
from fake_useragent import UserAgent
from pytrends.request import TrendReq
import signal
import pickle
import hashlib
import json
from datetime import datetime, timedelta
from serpapi.google_search import GoogleSearch
import wikipedia
from typing import List, Dict, Any, Optional, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables from .env file
load_dotenv()

# Global timeout settings
MAX_TOTAL_TIME = 500  # Increased to 500 seconds (8+ minutes) for extremely deep results
MAX_CONTENT_FETCH_TIME = 300  # Increased to 300 seconds (5 minutes)
MIN_SOURCES_REQUIRED = 10  # Minimum sources needed before analysis
MIN_URLS_TO_COLLECT = 50  # Ideal number of URLs to collect
MAX_URLS_TO_COLLECT = 150  # Maximum URLs to collect
MIN_ANALYZED_SOURCES = 50  # Minimum analyzed sources to include
MAX_ANALYZED_SOURCES = 150  # Maximum analyzed sources to include

# Handler for timeouts
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

class DeepResearch:
    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        # SerpAPI key for reliable search
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.ua = UserAgent()
        # Rotate user agents for each request to prevent rate limiting
        def get_headers():
            return {'User-Agent': self.ua.random}
        self.get_headers = get_headers
        
        # Initialize the sentence transformer model
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia('deep-research-tool/1.0', 'en')
        
        # Initialize PyTrends
        self.pytrends = TrendReq(hl='en-US', tz=360)
        
        # Cache for search results and analyses
        self.cache_file = ".research_cache.pkl"
        self.cache = self._load_cache()
        
        # Quality domains - variety of trusted sources (expanded beyond just academic/news)
        self.high_quality_domains = [
            # General information
            'wikipedia.org', 'britannica.com', 'howstuffworks.com', 'nationalgeographic.com',
            # Science and tech
            'nature.com', 'science.org', 'nasa.gov', 'esa.int', 'ieee.org', 'acm.org',
            'techcrunch.com', 'wired.com', 'theverge.com', 'cnet.com', 'zdnet.com',
            # News
            'bbc.com', 'reuters.com', 'apnews.com', 'npr.org', 
            # Industry
            'mckinsey.com', 'gartner.com', 'forrester.com', 'ibm.com', 'microsoft.com',
            # Entertainment
            'imdb.com', 'rottentomatoes.com', 'metacritic.com',
            # Shopping/Reviews
            'consumerreports.org', 'wirecutter.com', 'rtings.com',
            # Finance
            'bloomberg.com', 'investopedia.com', 'fool.com',
            # Health
            'nih.gov', 'mayoclinic.org', 'webmd.com', 'healthline.com'
        ]
        
        # Academic and research paper domains
        self.academic_domains = [
            'arxiv.org', 'researchgate.net', 'academia.edu', 'ssrn.com',
            'springer.com', 'sciencedirect.com', 'tandfonline.com', 'wiley.com',
            'ncbi.nlm.nih.gov', 'pubmed.gov', 'jstor.org', 'ieee.org'
        ]
        
        # Blacklisted domains - known low-quality or unreliable sources
        self.blacklisted_domains = [
            'quora.com', 'reddit.com', 'pinterest.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'tiktok.com', 'tumblr.com', 'medium.com',
            'blogspot.com', 'wordpress.com', 'wixsite.com', 'substack.com',
            'answers.yahoo.com', 'answers.com', 'wikihow.com', 
            'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com'
        ]
        
        # Content verification flags - terms that suggest questionable content
        self.content_flags = [
            'conspiracy', 'hoax', 'fake news', 'pseudoscience', 'debunked',
            'miracle cure', 'one weird trick', 'you won\'t believe', 'shocking truth'
        ]
        
    def _load_cache(self):
        """Load the cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    # Clean expired cache entries (older than 1 day)
                    now = datetime.now()
                    for key in list(cache.keys()):
                        if 'timestamp' in cache[key]:
                            if now - cache[key]['timestamp'] > timedelta(days=1):
                                del cache[key]
                    return cache
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save the cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except:
            pass
    
    def _cache_key(self, query):
        """Generate a cache key for a query"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get_specialized_sources(self, query):
        """Get specialized sources based on query keywords (shortened)"""
        # This method is deprecated as we are no longer using hardcoded sources
        # Now using dynamic search instead
        return []
    
    def google_search(self, query, num_results=5):
        """Perform a Google search for the query"""
        cache_key = f"google:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        urls = []
        try:
            # Using SerpAPI for reliable Google search results
            print(f"Performing SerpAPI Google search for: {query}")
            
            # Build the search parameters
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": min(num_results * 2, 100),  # Request more results than needed to account for filtering
                "gl": "us",  # Set to US locale for consistent results
            }
            
            # Make the API request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)
            
            if response.status_code != 200:
                print(f"SerpAPI request failed with status code: {response.status_code}")
                print(f"Error message: {response.text}")
                raise Exception(f"SerpAPI request failed: {response.text}")
            
            # Parse the JSON response
            search_results = response.json()
            
            # Extract organic results
            if "organic_results" in search_results:
                for result in search_results["organic_results"][:num_results]:
                    if "link" in result:
                        urls.append(result["link"])
            
            # If not enough results, also try to get featured snippet or answer box
            if len(urls) < num_results and "answer_box" in search_results and "link" in search_results["answer_box"]:
                urls.append(search_results["answer_box"]["link"])
                
            # Also try knowledge graph if present
            if len(urls) < num_results and "knowledge_graph" in search_results and "source" in search_results["knowledge_graph"]:
                source_info = search_results["knowledge_graph"]["source"]
                if isinstance(source_info, dict) and "link" in source_info:
                    urls.append(source_info["link"])
            
            print(f"Found {len(urls)} results from SerpAPI Google search")
            
            self.cache[cache_key] = urls
            self._save_cache()
        except Exception as e:
            print(f"SerpAPI Google search failed: {str(e)}")
            
            # Fallback to Wikipedia as a reliable source if search fails
            try:
                print("SerpAPI search failed, trying Wikipedia as fallback...")
                wiki_urls = self.search_wikipedia(query, num_results)
                urls.extend(wiki_urls)
                print(f"Found {len(urls)} results from Wikipedia fallback")
            except Exception as retry_e:
                print(f"Wikipedia fallback search failed: {str(retry_e)}")
        
        return urls
    
    def direct_access_wikipedia(self, title):
        """Directly access a Wikipedia page by title"""
        try:
            # First, try exact title
            print(f"Trying to access Wikipedia page: {title}")
            page = self.wiki.page(title)
            if page.exists():
                return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            
            # Try with SpaceX prefix
            spacex_title = f"SpaceX {title}"
            page = self.wiki.page(spacex_title)
            if page.exists():
                return f"https://en.wikipedia.org/wiki/{spacex_title.replace(' ', '_')}"
            
            # Try other variations
            variations = [
                title.replace("Flight", "flight"),
                title.replace("flight", "Flight"),
                f"Starship {title}" if "Starship" not in title else title,
                f"SpaceX Starship {title}" if "Starship" in title and "SpaceX" not in title else title
            ]
            
            for var in variations:
                page = self.wiki.page(var)
                if page.exists():
                    return f"https://en.wikipedia.org/wiki/{var.replace(' ', '_')}"
            
            return None
        except Exception as e:
            print(f"Error accessing Wikipedia: {str(e)}")
            return None
        
    def direct_content_fetch(self, urls, query="", max_urls=8):
        """Directly fetch content from a limited list of URLs with timeout"""
        contents = []
        # Score, validate, and prioritize URLs before fetching
        scored_urls = self._validate_and_score_urls(urls, query)
        
        # Separate high-quality, mid-quality and general sites
        high_quality_urls = []
        mid_quality_urls = []
        general_urls = []
        
        for score, url in scored_urls:
            domain = self._get_domain(url)
            if any(quality_domain in domain for quality_domain in self.high_quality_domains) or '.edu' in domain or '.gov' in domain:
                high_quality_urls.append((score, url))
            elif score >= 1.0:  # Mid-quality threshold
                mid_quality_urls.append((score, url))
            elif not any(blacklisted in domain for blacklisted in self.blacklisted_domains):
                general_urls.append((score, url))
        
        # Increased numbers for more comprehensive research
        high_quality_to_process = [url for _, url in high_quality_urls[:15]]  # Increased from 10
        mid_quality_to_process = [url for _, url in mid_quality_urls[:25]]    # Added mid-quality tier
        general_to_process = [url for _, url in general_urls[:30]]            # Increased from 20
        
        # Combine all lists for maximum coverage
        urls_to_process = high_quality_to_process + mid_quality_to_process + general_to_process
        
        print(f"Starting content fetch for {len(urls_to_process)} URLs ({len(high_quality_to_process)} high-quality, {len(mid_quality_to_process)} mid-quality, {len(general_to_process)} general sources)...")
        
        # Check if we have any URLs to process
        if len(urls_to_process) == 0:
            print("No URLs to process after filtering, returning empty result")
            return []
        
        # Set timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(MAX_CONTENT_FETCH_TIME)
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_urls, len(urls_to_process))) as executor:
                future_to_url = {executor.submit(self._fetch_content, url): url for url in urls_to_process}
                for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls_to_process), desc="Fetching content"):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content:
                            contents.append((url, content))
                            # Add an evaluation of content quality
                            if self._evaluate_content_quality(content) < 0.5:
                                print(f"Low quality content from {url}, will prioritize better sources")
                    except Exception as e:
                        print(f"Error processing {url}: {str(e)}")
        except TimeoutError:
            print("Content fetching timed out, proceeding with what we have...")
        finally:
            # Cancel the alarm
            signal.alarm(0)
            
        # Hardcoded backup for completely empty results
        if not contents and query:
            # If we have a query but no contents, get general info from Wikipedia as last resort
            print("No contents available. Trying to get general information from Wikipedia...")
            try:
                # Use first word of query to get general info
                main_topic = query.split()[0]
                wiki_url = f"https://en.wikipedia.org/wiki/{main_topic.capitalize()}"
                contents = [(wiki_url, self._fetch_content(wiki_url))]
                print(f"Added fallback Wikipedia content: {wiki_url}")
            except Exception as e:
                print(f"Failed to add fallback Wikipedia content: {str(e)}")
        
        return contents

    def _fetch_content(self, url):
        """Helper method to fetch content from a single URL with short timeout"""
        try:
            headers = self.get_headers()  # Get fresh headers for each request
            response = requests.get(url, headers=headers, timeout=10)  # Increased timeout
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Special handling for Wikipedia
                if 'wikipedia.org' in url:
                    content_div = soup.find('div', {'id': 'mw-content-text'})
                    if content_div:
                        # Get main content paragraphs
                        paragraphs = content_div.find_all('p')
                        text = ' '.join(p.get_text() for p in paragraphs)
                    else:
                        text = soup.get_text()
                else:
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                        element.decompose()
                    
                    # Try to find the main content
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'}) or soup.find('div', {'class': 'content'})
                    
                    if main_content:
                        paragraphs = main_content.find_all('p')
                        text = ' '.join(p.get_text() for p in paragraphs)
                    else:
                        paragraphs = soup.find_all('p')
                        text = ' '.join(p.get_text() for p in paragraphs)
                
                # Clean text
                text = re.sub(r'\s+', ' ', text).strip()
                text = re.sub(r'\[\d+\]', '', text)  # Remove citation marks
                
                if text and len(text) > 100:
                    # Capture more content from each page for maximum detail (increased from 2500)
                    return f"Source: {url}\n\n{text[:4000]}"
            
            return None
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
        
    def search_wikipedia(self, query, num_results=3):
        """Improved Wikipedia search with better fallbacks"""
        print(f"Searching Wikipedia...")
        cache_key = f"wiki:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        urls = []
        
        # Check for Starship flight matches first
        flight_match = re.search(r'(starship\s*flight\s*\d+)', query.lower())
        if flight_match:
            # Try direct access first
            flight_title = flight_match.group(1).title()
            wiki_url = self.direct_access_wikipedia(flight_title)
            if wiki_url:
                urls.append(wiki_url)
            
            # Also try SpaceX Starship general page
            wiki_url = self.direct_access_wikipedia("SpaceX Starship")
            if wiki_url and wiki_url not in urls:
                urls.append(wiki_url)
                
            # And the flight test program page
            wiki_url = self.direct_access_wikipedia("SpaceX Starship flight test program")
            if wiki_url and wiki_url not in urls:
                urls.append(wiki_url)
        else:
            # Try direct page access
            page = self.wiki.page(query)
            if page.exists():
                urls.append(f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}")
            
            # Try to find related pages by keywords
            keywords = query.split()
            for keyword in keywords:
                if len(keyword) > 3:  # Only use meaningful keywords
                    page = self.wiki.page(keyword.capitalize())
                    if page.exists():
                        url = f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}"
                        if url not in urls:
                            urls.append(url)
        
        if urls:
            # If we found pages, also check links on the first page
            try:
                page_title = urls[0].split('/')[-1].replace('_', ' ')
                page = self.wiki.page(page_title)
                if page.exists():
                    for title, _ in list(page.links.items())[:5]:
                        if "Category:" not in title and "Template:" not in title:
                            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                            if url not in urls:
                                urls.append(url)
            except Exception as e:
                print(f"Error getting Wikipedia links: {str(e)}")
            
        self.cache[cache_key] = urls
        self._save_cache()
        
        print(f"Found {len(urls)} Wikipedia pages")
        return urls
    
    def search_duckduckgo(self, query, num_results=5):
        """Quick DuckDuckGo search with fewer results"""
        cache_key = f"ddg:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        urls = []
        try:
            # Using SerpAPI for DuckDuckGo search results instead of direct approach
            print(f"Performing SerpAPI DuckDuckGo search for: {query}")
            
            # Build the search parameters
            params = {
                "engine": "duckduckgo",
                "q": query,
                "api_key": self.serpapi_key,
            }
            
            # Make the API request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)
            
            if response.status_code != 200:
                print(f"SerpAPI DuckDuckGo request failed with status code: {response.status_code}")
                raise Exception(f"SerpAPI DuckDuckGo request failed: {response.text}")
            
            # Parse the JSON response
            search_results = response.json()
            
            # Extract organic results (the structure for DuckDuckGo via SerpAPI)
            if "organic_results" in search_results:
                for result in search_results["organic_results"][:num_results]:
                    if "link" in result:
                        urls.append(result["link"])
            
            print(f"Found {len(urls)} results from SerpAPI DuckDuckGo search")
            
            self.cache[cache_key] = urls
            self._save_cache()
        except Exception as e:
            print(f"SerpAPI DuckDuckGo search failed: {str(e)}")
            
            # If SerpAPI fails, try a fallback to Wikipedia
            try:
                print("SerpAPI DuckDuckGo search failed, falling back to Wikipedia...")
                wiki_urls = self.search_wikipedia(query, num_results)
                urls.extend(wiki_urls)
                print(f"Found {len(urls)} results from Wikipedia fallback")
            except Exception as wiki_e:
                print(f"Wikipedia fallback search failed: {str(wiki_e)}")
            
        return urls
    
    def search_and_collect_urls(self, query, num_results=10):
        """Quick search for fewer URLs"""
        print(f"Searching for: {query}")
        
        all_urls = set()
        
        # First try DuckDuckGo search as primary source (skipping Google as requested)
        print("Performing DuckDuckGo search as primary source...")
        ddg_urls = self.search_duckduckgo(query, 15)  # Try to get a reasonable number with SerpAPI
        all_urls.update(ddg_urls)
        
        # Always add Wikipedia for reliability
        print("Adding Wikipedia source for reliability...")
        wiki_urls = self.search_wikipedia(query, 3)
        all_urls.update(wiki_urls)
        
        # If DuckDuckGo failed, try with Google as a fallback
        if len(all_urls) == 0:
            print("DuckDuckGo search failed to return results, trying Google as fallback...")
            try:
                google_urls = self.google_search(query, 15)  # Using SerpAPI Google search
                all_urls.update(google_urls)
                print(f"Found {len(google_urls)} results from Google fallback")
            except Exception as e:
                print(f"Google fallback search failed: {str(e)}")
        
        # Check if we found any URLs
        if len(all_urls) == 0:
            print("No URLs found from primary search, trying more variations")
            # Try more variations immediately
            variations = [
                f"{query} guide", 
                f"what is {query}",
                f"{query} meaning",
                f"learn about {query}",
                f"{query} facts"
            ]
            
            for var_query in variations:
                print(f"Trying variation: {var_query}")
                var_urls = self.search_duckduckgo(var_query, 15)
                if var_urls:
                    all_urls.update(var_urls)
                    print(f"Found {len(var_urls)} URLs with variation '{var_query}'")
                    break
        
        # Try multiple search variations for more comprehensive results
        search_variations = [
            f"{query} information", 
            f"{query} meaning",
            f"{query} definition"
        ]
        
        # Try Google for these variations first to avoid rate limits
        for var_query in search_variations:
            try:
                var_urls = self.google_search(var_query, 10)
                if var_urls:
                    all_urls.update(var_urls)
                    print(f"Added {len(var_urls)} Google sources for '{var_query}'")
                    if len(var_urls) > 5:
                        break
            except Exception as e:
                print(f"Google variation search failed: {str(e)}")
        
        # Then try DuckDuckGo with more spacing and smaller batches
        for var_query in search_variations:
            try:
                time.sleep(3)  # Moderate delay
                var_urls = self.search_duckduckgo(var_query, 5)  # Try to get a few results
                if var_urls:
                    all_urls.update(var_urls)
                    print(f"Added {len(var_urls)} DuckDuckGo sources from '{var_query}'")
                    # If we found good results, no need to try all variations
                    if len(var_urls) > 3:
                        break
            except Exception as e:
                print(f"DuckDuckGo variation search failed: {str(e)}")
        
        # Try multiple news searches with variations
        news_variations = [
            f"{query} news",
            f"{query} update"
        ]
        
        for news_query in news_variations:
            try:
                # Use SerpAPI's Google News search for more reliable news results
                news_urls = self.search_news(news_query, 8)
                all_urls.update(news_urls)
                
                # Break if we found good results
                if len(news_urls) > 3:
                    print(f"Added {len(news_urls)} news sources from '{news_query}'")
                    break
            except Exception as e:
                print(f"News search failed: {str(e)}")
        
        # If we still don't have enough sources, try some more targeted approaches
        if len(all_urls) < MIN_URLS_TO_COLLECT:
            print(f"Only found {len(all_urls)} URLs so far. Trying Wikipedia and academic sources...")
            
            # Extract only important keywords to focus search better
            keywords = [word for word in query.split() 
                       if word.lower() not in {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of', 'what', 'how', 'why', 'when', 'where', 'who'}
                       and len(word) > 3]
            for keyword in keywords:
                wiki_urls = self.search_wikipedia(keyword, 2)
                all_urls.update(wiki_urls)
            
            # Try specialized databases based on query topic - use Google as it's more reliable
            try:
                for domain in self.academic_domains[:3]:
                    site_query = f"site:{domain} {query}"
                    specialized_urls = self.google_search(site_query, 5)
                    all_urls.update(specialized_urls)
            except Exception as e:
                print(f"Academic search failed: {str(e)}")
            
            # Last attempt with very simple query via Google
            try:
                simplified_query = " ".join(query.split()[:2])
                print(f"Trying simplified query: {simplified_query}")
                more_ddg_urls = self.search_duckduckgo(simplified_query, 25)
                all_urls.update(more_ddg_urls)
            except Exception:
                pass
        
        # Score and sort URLs by relevance and quality
        scored_urls = self._score_urls(list(all_urls), query)
        
        # Separate high-quality and lower-quality sources
        high_quality = []
        lower_quality = []
        
        for score, url in scored_urls:
            domain = self._get_domain(url)
            # Determine if this is a high-quality source
            if (any(quality_domain in domain for quality_domain in self.high_quality_domains) or 
                any(academic_domain in domain for academic_domain in self.academic_domains) or
                '.gov' in domain or '.edu' in domain):
                high_quality.append(url)
            elif not any(blacklisted in domain for blacklisted in self.blacklisted_domains):
                lower_quality.append(url)
        
        # Take more URLs in each category for maximum coverage
        high_quality = high_quality[:75]   # Increased from 50
        lower_quality = lower_quality[:150]  # Increased from 100
        
        # Combine URLs with high-quality ones first
        urls = high_quality + lower_quality
        
        # Ensure we don't exceed the maximum
        if len(urls) > MAX_URLS_TO_COLLECT + 100:  # Doubled the increase to get even more diverse sources
            urls = urls[:MAX_URLS_TO_COLLECT + 100]
            
        print(f"Selected {len(high_quality)} high-quality sources and {len(lower_quality)} diverse sources for a total of {len(urls)} URLs")

        if not urls:
            print("Warning: No URLs were found.")
            urls = ["https://en.wikipedia.org/wiki/Main_Page"]

        print(f"Found {len(urls)} unique URLs")
        return urls

    def analyze_with_groq(self, query, contents):
        """Analyze collected content using Groq's model with research paper-level accuracy"""
        if not contents:
            return "Sorry, I couldn't find reliable information on this topic."
            
        # Safety check - ensure we have at least one source
        if len(contents) == 0:
            return "Sorry, I couldn't find any reliable sources for this query. Please try a different search term."
            
        # Deeply evaluate, validate and prioritize content by relevance and quality
        print("Performing comprehensive content validation and relevance scoring...")
        validated_contents = self._deeply_validate_contents(query, contents)
        
        # Separate high and lower quality sources
        high_quality = []
        lower_quality = []
        
        for item in validated_contents:
            url, content, quality_score = item
            if quality_score >= 7.5:  # Increased threshold for high quality
                high_quality.append(item)
            else:
                lower_quality.append(item)
        
        # Select top sources from each category
        top_high_quality = high_quality[:15]  # Reduced from 20
        top_lower_quality = lower_quality[:20]  # Reduced from 30
        
        # Group content by quality score but with more granular categories
        top_tier = []
        mid_tier = []
        general_tier = []
        
        for item in validated_contents:
            url, content, quality_score = item
            if quality_score >= 7.5:  # Increased threshold for high reliability
                top_tier.append(item)
            elif quality_score >= 5.5:  # Increased threshold for medium reliability
                mid_tier.append(item)
            else:  # Lower reliability but still valuable
                general_tier.append(item)
        
        # Take a more balanced approach with different types of content
        # Adjust based on query type (check if educational, news, etc.)
        is_educational_query = any(term in query.lower() for term in ['history', 'science', 'math', 'learn', 'study', 'education'])
        is_current_events = any(term in query.lower() for term in ['news', 'current', 'recent', 'latest', 'today', 'update'])
        
        if is_educational_query:
            # For educational queries, favor high-quality sources
            selected_top = top_tier[:15]  # Reduced from 20
            selected_mid = mid_tier[:12]  # Reduced from 15
            selected_general = general_tier[:10]  # Reduced from 12
        elif is_current_events:
            # For news queries, include more diverse sources
            selected_top = top_tier[:12]  # Reduced from 15
            selected_mid = mid_tier[:15]  # Reduced from 20
            selected_general = general_tier[:12]  # Reduced from 15
        else:
            # For general queries, keep a wide balance
            selected_top = top_tier[:12]  # Reduced from 15
            selected_mid = mid_tier[:15]  # Reduced from 20
            selected_general = general_tier[:15]  # Reduced from 20
        
        # Combine all selected contents
        selected_contents = selected_top + selected_mid + selected_general
        
        print(f"Analyzing {len(selected_contents)} sources ({len(top_high_quality)} high-quality, {len(top_lower_quality)} diverse sources)...")
        
        # Keep only the most relevant sources without grouping by domain to save tokens
        context_parts = []
        source_index = 1
        
        for url, content, quality_score in selected_contents:
            # Trim content more aggressively (reduced from 4000 to 2000 chars per source)
            trimmed_content = content[:2000] if content else ""
            
            # Add trimmed content with source information
            context_parts.append(f"Source {source_index}: {url} [Quality Score: {quality_score:.1f}/10]\n{trimmed_content}")
            source_index += 1
            
            # Stop adding sources when we're approaching token limit
            total_chars = sum(len(part) for part in context_parts)
            if total_chars > 15000:  # Reduced from 25000 (~3750 tokens)
                print(f"Reached character limit at {source_index-1} sources ({total_chars} chars)")
                break
        context = "\n\n---\n\n".join(context_parts)

        # Estimate token count (rough approximation)
        estimated_tokens = len(context) / 4  # ~4 chars per token on average
        print(f"Estimated token count for context: ~{int(estimated_tokens)} tokens")

        # Create an enhanced system prompt for research paper-level analysis
        system_prompt = """You are an expert research assistant who delivers engaging, comprehensive research paper-level analysis with maximum accuracy and depth. Your key objectives:

1. Format your output with clear academic structure and engaging narrative:
   - Use # for main title (topic name)
   - Use ## for major section headings (Introduction, Methodology, Results, Discussion, Conclusion)
   - Use ### for subsections
   - Use **bold text** for key concepts, dates, and names
   - Use *italic text* for emphasis on key points or terminology
   - Properly format quotes as > blockquotes when citing direct text
   - Always put section headings on their own lines with space above and below
   - Write in an engaging, flowing narrative style that maintains reader interest

2. Organize content in a clear, academic manner with engaging flow:
   - Begin with a compelling introduction that hooks the reader
   - Include a comprehensive introduction section with background and context
   - Use logical sections and subsections for different aspects
   - End with a detailed conclusion section summarizing key findings
   - Maintain a narrative thread throughout the analysis

3. Write in academic style with maximum accuracy and engagement:
   - Use precise, formal language while maintaining readability
   - Support every claim with evidence
   - Present balanced perspectives
   - Acknowledge limitations and uncertainties
   - Cross-reference information between sources
   - Note when sources disagree and explain why
   - Use transitions to maintain flow between sections

4. Source citation and evaluation:
   - Cite ALL facts using **[Source X]** format (bold)
   - Assess each source for reliability using Quality Score
   - Cross-reference information between multiple sources
   - Note when sources disagree and present multiple viewpoints
   - Prioritize high-quality sources (Quality Score >= 7.5)

5. Ensure maximum accuracy and depth:
   - Double-check all facts and figures
   - Verify dates and timelines
   - Cross-reference statistics and data
   - Present multiple perspectives when available
   - Acknowledge uncertainties and limitations
   - Include methodological details when available

6. For predictive queries:
   - State assumptions clearly
   - Present multiple scenarios
   - Support with historical data
   - Acknowledge uncertainty
   - Explain reasoning

7. Include comprehensive analysis:
   - Historical context
   - Current state
   - Future implications
   - Multiple perspectives
   - Methodological details

8. Write detailed, comprehensive paragraphs:
   - Each paragraph should be at least 5-7 sentences long
   - Include specific details, examples, and evidence
   - Connect ideas logically between paragraphs
   - Use transitions to maintain flow
   - Provide context and background for complex concepts

9. Cite ALL facts using **[Source X]** format (bold)
10. Cross-reference information between sources
11. End with a detailed conclusion
12. Add a comprehensive summary table at the end with key findings

You have {len(context_parts)} diverse sources available, each with quality scores (0-10).

Sources:
{context}"""

        # Create an enhanced user prompt for research paper-level analysis
        user_prompt = f"""# Comprehensive Research Analysis: {query}

I need an engaging, research paper-level analysis of this topic with maximum accuracy and depth. Your response should:

1. Begin with a compelling main title (# heading) and introduction that hooks the reader
2. Organize content with clear ## section headings and ### subsection headings
3. Use proper academic formatting:
   - **Bold** for key terms, dates, and concepts
   - *Italics* for emphasis and terminology
   - > Blockquotes for direct quotations
   - Leave space above and below headings

4. Write in academic style with maximum accuracy and engagement:
   - Support every claim with evidence
   - Present balanced perspectives
   - Acknowledge limitations
   - Cross-reference information
   - Maintain a flowing narrative style

5. For predictive queries:
   - State assumptions clearly
   - Present multiple scenarios
   - Support with historical data
   - Acknowledge uncertainty
   - Explain reasoning

6. Include comprehensive analysis:
   - Historical context
   - Current state
   - Future implications
   - Multiple perspectives
   - Methodological details

7. Cite ALL facts using **[Source X]** format (bold)
8. Cross-reference information between sources
9. End with a detailed conclusion
10. Add a comprehensive summary table at the end with:
    - Key Findings
    - Main Arguments
    - Supporting Evidence
    - Areas of Uncertainty
    - Future Implications

You have {len(context_parts)} diverse sources available, each with quality scores (0-10).

Sources:
{context}"""

        # Prepare messages with more concise prompts
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use a more appropriate model that can handle our request within limits
        model = "deepseek-r1-distill-qwen-32b"  # Using a model with larger context window
        
        try:
            # Set absolute maximum token limit for an exhaustive response
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages, 
                temperature=0.7,   # Slightly reduced for more focused analysis
                max_tokens=8000    # Increased token limit for more comprehensive response
            )
            
            # Cache the results
            cache_key = self._cache_key(query)
            self.cache[cache_key] = {
                'analysis': completion.choices[0].message.content,
                'timestamp': datetime.now()
            }
            self._save_cache()
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in Groq API call: {str(e)}")
            return f"I encountered an error while analyzing the information: {str(e)}"

    def research(self, query):
        """Main research method with global timeout"""
        print("Starting comprehensive research process...")
        
        cache_key = self._cache_key(query)
        if cache_key in self.cache and 'analysis' in self.cache[cache_key]:
            cached_analysis = self.cache[cache_key]['analysis']
            error_phrases = ["couldn't find", "took too long", "error", "sorry, i couldn't"]
            if not any(phrase in cached_analysis.lower() for phrase in error_phrases):
                print("Found cached results!")
                return cached_analysis
            else:
                del self.cache[cache_key]
                self._save_cache()
        
        # Set global timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(MAX_TOTAL_TIME)
        
        try:
            # Step 1: Collect information from all sources
            print("Collecting information from multiple sources...")
            
            # Get Wikipedia results
            wiki_results = self.search_wikipedia_direct(query)
            print(f"Found {len(wiki_results)} Wikipedia results")
            
            # Get YouTube results
            youtube_results = self.search_youtube(query)
            print(f"Found {len(youtube_results)} YouTube results")
            
            # Get Google Maps results if query seems location-based
            location_keywords = ['near', 'in', 'at', 'around', 'close to', 'location', 'place', 'restaurant', 'cafe', 'store']
            if any(keyword in query.lower() for keyword in location_keywords):
                maps_results = self.search_google_maps(query)
                print(f"Found {len(maps_results)} Google Maps results")
            else:
                maps_results = []
            
            # Step 2: Search and collect URLs (existing functionality)
            urls = self.search_and_collect_urls(query)
            
            if isinstance(urls, str):
                return urls
                
            if not urls:
                return "Sorry, I couldn't find any relevant sources for your query. Please try a different search term or phrase your query differently."
            
            # Step 3: Fetch content from sources
            print(f"Fetching content from up to {min(len(urls), 60)} sources for comprehensive analysis...")
            contents = self.direct_content_fetch(urls, query, max_urls=60)
            
            # Step 4: Prepare comprehensive context including new sources
            context_parts = []
            
            # Add Wikipedia content
            for wiki_result in wiki_results:
                context_parts.append(f"Wikipedia Source: {wiki_result['url']}\n{wiki_result['content']}")
            
            # Add YouTube content
            if youtube_results:
                context_parts.append("\nRelevant YouTube Videos:")
                for video in youtube_results:
                    context_parts.append(f"Video: {video['title']}\nChannel: {video['channel']}\nURL: {video['link']}")
            
            # Add Google Maps content
            if maps_results:
                context_parts.append("\nRelevant Locations:")
                for place in maps_results:
                    context_parts.append(f"Place: {place['title']}\nAddress: {place['address']}\nType: {place['type']}\nDescription: {place['description']}")
            
            # Add web content
            for url, content in contents:
                context_parts.append(f"Web Source: {url}\n{content}")
            
            # Step 5: Analyze with Groq
            analysis = self.analyze_with_groq(query, contents)
            
            return analysis
            
        except TimeoutError:
            return "The research process took too long and was automatically stopped. Please try a more specific query."
        finally:
            # Cancel the alarm
            signal.alarm(0)

    # Add new methods for quality and relevance scoring
    def _score_urls(self, urls, query=""):
        """Score URLs by domain quality, relevance, and other factors"""
        scored_urls = []
        
        for url in urls:
            score = 0
            domain = self._get_domain(url)
            
            # Boost score for high-quality domains
            if any(quality_domain in domain for quality_domain in self.high_quality_domains):
                score += 4  # Increased weight for high-quality domains for accuracy
                
            # Extra boost for academic sources
            if any(academic_domain in domain for academic_domain in self.academic_domains):
                score += 4  # Increased weight for academic sources for accuracy
            
            # Penalize potentially low-quality domains
            if 'blog.' in domain or '.blog.' in domain:
                score -= 2  # More penalty for blogs as less accurate
                
            # Prefer HTTPS for security
            if url.startswith('https://'):
                score += 0.5
                
            # Prefer shorter, cleaner URLs (often more authoritative)
            if url.count('/') < 5:
                score += 1
                
            # Prefer non-forum content as it's often more reliable
            if 'forum' in url or 'discussion' in url:
                score -= 2  # More penalty for forums as less accurate
            
            # Prefer educational and government sites for accuracy
            if '.edu' in domain or '.gov' in domain:
                score += 3  # Higher boost for proven accurate sources
            
            # Prefer established news sources
            if any(news_domain in domain for news_domain in ['reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 'republicworld.com', 'foxnews.com']):
                score += 2  # Boost for reputable news sources
            
            # Strongly favor URLs containing query terms
            query_keywords = [k.lower() for k in query.split() if len(k) > 3 and k.lower() not in {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}]
            for keyword in query_keywords:
                if keyword in url.lower():
                    score += 3  # Increased boost for directly relevant URLs
            
            scored_urls.append((score, url))
        
        # Sort by score descending
        return sorted(scored_urls, key=lambda x: x[0], reverse=True)
    
    def _get_domain(self, url):
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return ""

    def _evaluate_content_quality(self, content):
        """Basic evaluation of content quality"""
        # This is a simple heuristic - could be improved with ML
        if not content:
            return 0
        
        score = 1.0
        
        # Length is a basic quality indicator
        text_length = len(content)
        if text_length < 1000:  # Increased minimum length threshold
            score *= 0.5
        elif text_length > 8000:  # Increased maximum length threshold
            score *= 1.5
            
        # Check for indicators of quality content
        if "research" in content.lower() or "study" in content.lower():
            score *= 1.2
            
        # Check for citations which suggest academic quality
        if "[" in content and "]" in content:
            score *= 1.3
            
        # Check for dates which suggest currency
        if re.search(r'202[0-9]', content):  # Recent dates
            score *= 1.2
            
        # Check for detailed explanations and analysis
        if any(term in content.lower() for term in ['analysis', 'explanation', 'detailed', 'comprehensive', 'in-depth']):
            score *= 1.2
            
        # Check for data and statistics
        if re.search(r'\d+%|\d+\.\d+%', content) or re.search(r'figure \d+|table \d+', content.lower()):
            score *= 1.2
            
        return min(score, 2.0)  # Cap at 2.0
        
    def _prioritize_contents(self, query, contents):
        """Prioritize contents based on relevance to query and quality"""
        if not contents:
            return []
            
        # Convert query to lowercase for comparison
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        scored_contents = []
        for url, content in contents:
            # Start with base content quality score
            score = self._evaluate_content_quality(content)
            
            # Check domain quality
            domain = self._get_domain(url)
            if any(quality_domain in domain for quality_domain in self.high_quality_domains):
                score += 2
                
            # Calculate relevance by term matching
            content_lower = content.lower()
            matching_terms = sum(1 for term in query_terms if term in content_lower)
            relevance_score = matching_terms / max(1, len(query_terms))
            score += relevance_score * 3  # Weight relevance highly
            
            # Check if exact query appears
            if query_lower in content_lower:
                score += 2
                
            scored_contents.append((score, (url, content)))
            
        # Sort by score descending and return just the contents
        return [content for _, content in sorted(scored_contents, key=lambda x: x[0], reverse=True)]

    def _validate_and_score_urls(self, urls, query=""):
        """Score URLs by domain quality, relevance, and other factors with validation"""
        scored_urls = []
        
        for url in urls:
            # Skip obviously invalid URLs
            if not url.startswith('http'):
                continue
                
            # Extract domain
            domain = self._get_domain(url)
            if not domain:
                continue
                
            # Skip blacklisted domains
            if any(blacklisted in domain for blacklisted in self.blacklisted_domains):
                print(f"Skipping blacklisted domain: {domain}")
                continue
            
            # Start scoring
            score = 0
            
            # Boost score for high-quality domains
            if any(quality_domain in domain for quality_domain in self.high_quality_domains):
                score += 4  # Increased weight for high-quality sources
                
            # Extra boost for academic sources
            if any(academic_domain in domain for academic_domain in self.academic_domains):
                score += 4  # Increased weight for academic sources
            
            # Penalize potentially low-quality domains
            if any(term in domain for term in ['blog', 'forum', 'discussion', 'community']):
                score -= 2  # Increased penalty for less reliable sources
                
            # Prefer HTTPS for security
            if url.startswith('https://'):
                score += 0.5
                
            # Strongly favor URLs containing query terms
            query_keywords = [k.lower() for k in query.split() if len(k) > 3 and k.lower() not in {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}]
            for keyword in query_keywords:
                if keyword in url.lower():
                    score += 3  # Increased boost for directly relevant URLs
            
            # Prefer shorter, cleaner URLs (often more authoritative)
            if url.count('/') < 5:
                score += 1
            elif url.count('/') > 8:
                score -= 1  # Increased penalty for deep URLs
                
            # Prefer government, educational or organizational domains
            if any(suffix in domain for suffix in ['.gov', '.edu', '.org']):
                score += 3  # Increased boost for authoritative domains
                
            # Prefer domains with clear topics that match keywords
            for keyword in query.lower().split():
                if len(keyword) > 3 and keyword in domain:
                    score += 2  # Increased boost for topic-specific domains
            
            # Additional checks for research paper quality
            if any(term in url.lower() for term in ['research', 'study', 'analysis', 'paper', 'publication']):
                score += 2  # Boost for research-focused content
                
            if any(term in url.lower() for term in ['methodology', 'method', 'procedure', 'experiment']):
                score += 2  # Boost for methodological content
                
            if any(term in url.lower() for term in ['data', 'statistics', 'results', 'findings']):
                score += 2  # Boost for data-driven content
                
            if any(term in url.lower() for term in ['conclusion', 'summary', 'discussion']):
                score += 1  # Boost for analytical content
            
            scored_urls.append((score, url))
        
        # Sort by score descending
        return sorted(scored_urls, key=lambda x: x[0], reverse=True)

    def _deeply_validate_contents(self, query, contents):
        """Perform deep validation and scoring of content relevance and quality"""
        if not contents:
            return []
            
        print(f"Validating {len(contents)} sources for quality and relevance...")
        
        # Convert query to lowercase for comparison
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Remove stop words from query terms to focus on meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}
        meaningful_terms = {term for term in query_terms if term not in stop_words and len(term) > 2}
        
        validated_contents = []
        for url, content in contents:
            # Skip empty content
            if not content or len(content) < 300:  # Increased minimum length
                continue
                
            # Start with base content quality evaluation
            quality_score = self._evaluate_content_quality(content)
            
            # Check for content flags that indicate potential misinformation
            content_lower = content.lower()
            if any(flag in content_lower for flag in self.content_flags):
                quality_score *= 0.3  # More severe penalty for flagged content
                
            # Check domain quality
            domain = self._get_domain(url)
            if any(quality_domain in domain for quality_domain in self.high_quality_domains):
                quality_score += 3  # Increased boost for high-quality domains
            
            # Calculate relevance by term matching (focusing on meaningful terms)
            matching_meaningful_terms = sum(1 for term in meaningful_terms if term in content_lower)
            if meaningful_terms:
                relevance_score = matching_meaningful_terms / len(meaningful_terms)
                quality_score += relevance_score * 8  # Increased weight for relevance
            
            # Check for exact phrases from query (stronger signal than just terms)
            query_words = query_lower.split()
            for i in range(len(query_words) - 1):
                phrase = ' '.join(query_words[i:i+2])
                if len(phrase) > 5 and phrase in content_lower:
                    quality_score += 3  # Increased weight for exact phrase matches
                    
                if i < len(query_words) - 2:
                    phrase3 = ' '.join(query_words[i:i+3])
                    if phrase3 in content_lower:
                        quality_score += 4  # Increased weight for 3-word matches
            
            # Check for exact query match
            if query_lower in content_lower:
                quality_score += 6  # Increased weight for exact query match
                
            # Check for recent dates (currency of information)
            if re.search(r'202[3-4]', content):
                quality_score += 2
            elif re.search(r'202[0-2]', content):
                quality_score += 1
                
            # Additional research paper quality checks
            research_indicators = [
                'methodology', 'method', 'procedure', 'experiment',
                'data', 'statistics', 'results', 'findings',
                'conclusion', 'summary', 'discussion',
                'research', 'study', 'analysis', 'paper',
                'publication', 'journal', 'peer-reviewed'
            ]
            
            for indicator in research_indicators:
                if indicator in content_lower:
                    quality_score += 1.5  # Boost for research indicators
            
            # Check for citations and references
            if re.search(r'\[\d+\]', content) or re.search(r'\(\d{4}\)', content):
                quality_score += 2  # Boost for citations
                
            # Check for data presentation
            if re.search(r'\d+%|\d+\.\d+%', content) or re.search(r'figure \d+|table \d+', content.lower()):
                quality_score += 2  # Boost for data presentation
                
            # Normalize quality score to 0-10 scale
            normalized_score = min(10, quality_score)
            
            # Set higher threshold for research paper quality
            if normalized_score >= 3.0:  # Increased threshold for better quality
                validated_contents.append((url, content, normalized_score))
            
        # Sort by quality score descending
        return sorted(validated_contents, key=lambda x: x[2], reverse=True)
        
    def _group_by_domain(self, contents):
        """Group contents by domain to ensure source diversity"""
        domain_contents = {}
        
        for content_item in contents:
            url = content_item[0]
            domain = self._get_domain(url)
            
            if domain not in domain_contents:
                domain_contents[domain] = []
                
            domain_contents[domain].append(content_item)
            
        # Ensure we don't have too many sources from the same domain
        # by prioritizing the best content from each domain
        balanced_contents = {}
        for domain, items in domain_contents.items():
            # Sort items by quality score
            sorted_items = sorted(items, key=lambda x: x[2], reverse=True)
            
            # Take at most 3 best items from each domain to ensure diversity
            balanced_contents[domain] = sorted_items[:3]
            
        return balanced_contents

    # Add this new method for news search with SerpAPI
    def search_news(self, query, num_results=5):
        """Search for news articles using SerpAPI Google News engine"""
        cache_key = f"news:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        urls = []
        try:
            print(f"Performing SerpAPI News search for: {query}")
            
            # Build the search parameters
            params = {
                "engine": "google_news",
                "q": query,
                "api_key": self.serpapi_key,
                "num": min(num_results * 2, 50),  # Request more results than needed
            }
            
            # Make the API request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)
            
            if response.status_code != 200:
                print(f"SerpAPI News request failed with status code: {response.status_code}")
                raise Exception(f"SerpAPI News request failed: {response.text}")
            
            # Parse the JSON response
            search_results = response.json()
            
            # Extract news results
            if "news_results" in search_results:
                for result in search_results["news_results"][:num_results]:
                    if "link" in result:
                        urls.append(result["link"])
            
            print(f"Found {len(urls)} results from SerpAPI News search")
            
            self.cache[cache_key] = urls
            self._save_cache()
        except Exception as e:
            print(f"SerpAPI News search failed: {str(e)}")
        
        return urls

    def search_wikipedia_direct(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Direct Wikipedia search using the wikipedia module"""
        try:
            # Search Wikipedia
            search_results = wikipedia.search(query, results=num_results)
            wiki_results = []
            
            for title in search_results:
                try:
                    # Get the page
                    page = wikipedia.page(title)
                    
                    # Extract relevant information
                    result = {
                        'title': page.title,
                        'url': page.url,
                        'summary': page.summary,
                        'content': page.content[:2000]  # First 2000 chars of content
                    }
                    wiki_results.append(result)
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    for option in e.options[:3]:  # Take first 3 options
                        try:
                            page = wikipedia.page(option)
                            result = {
                                'title': page.title,
                                'url': page.url,
                                'summary': page.summary,
                                'content': page.content[:2000]
                            }
                            wiki_results.append(result)
                        except:
                            continue
                except:
                    continue
                    
            return wiki_results
        except Exception as e:
            print(f"Wikipedia direct search error: {str(e)}")
            return []

    def search_youtube(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search YouTube using SerpAPI"""
        try:
            # Build the search parameters
            params = {
                "engine": "youtube",
                "search_query": query,
                "api_key": self.serpapi_key,
                "num": num_results
            }
            
            # Make the API request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)
            
            if response.status_code != 200:
                print(f"SerpAPI YouTube request failed with status code: {response.status_code}")
                raise Exception(f"SerpAPI YouTube request failed: {response.text}")
            
            # Parse the JSON response
            search_results = response.json()
            
            video_results = []
            if "video_results" in search_results:
                for video in search_results["video_results"][:num_results]:
                    video_info = {
                        'title': video.get('title', ''),
                        'link': video.get('link', ''),
                        'thumbnail': video.get('thumbnail', ''),
                        'duration': video.get('duration', ''),
                        'views': video.get('views', ''),
                        'channel': video.get('channel', ''),
                        'description': video.get('description', '')
                    }
                    video_results.append(video_info)
            
            print(f"Found {len(video_results)} YouTube videos")
            return video_results
            
        except Exception as e:
            print(f"YouTube search error: {str(e)}")
            return []

    def search_google_maps(self, query: str, location: Optional[str] = None, radius: int = 5000) -> List[Dict[str, Any]]:
        """Search Google Maps using SerpAPI"""
        try:
            params = {
                "engine": "google_maps",
                "q": query,
                "api_key": self.serpapi_key,
                "radius": radius
            }
            
            # If location is provided, add it to params
            if location:
                params["ll"] = location
                
            search = GoogleSearch(params)
            results = search.get_dict()
            
            map_results = []
            if "local_results" in results:
                for place in results["local_results"]:
                    place_info = {
                        'title': place.get('title', ''),
                        'address': place.get('address', ''),
                        'phone': place.get('phone', ''),
                        'website': place.get('website', ''),
                        'rating': place.get('rating', ''),
                        'reviews': place.get('reviews', ''),
                        'type': place.get('type', ''),
                        'description': place.get('description', '')
                    }
                    map_results.append(place_info)
                    
            return map_results
        except Exception as e:
            print(f"Google Maps search error: {str(e)}")
            return []

def main():
    # Initialize the research system
    researcher = DeepResearch()
    
    # Get user query
    query = input("Enter your research query: ")
    
    # Record start time
    start_time = time.time()
    
    # Perform research
    result = researcher.research(query)
    
    # Format results for UI processing
    formatted_result = format_for_ui(result)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nResearch Results (completed in {elapsed_time:.2f} seconds):")
    print("-" * 50)
    print(formatted_result)
    
    # Save to text file
    filename = f"research_{hashlib.md5(query.encode()).hexdigest()[:8]}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(formatted_result)
    
    print(f"\nResults saved to {filename}")

def format_for_ui(text):
    """Format the research results for UI processing with enhanced markdown formatting"""
    # The model should already be generating good markdown, but we'll enhance it
    lines = text.split('\n')
    formatted_lines = []
    in_code_block = False
    previous_line_was_empty = False
    
    for i, line in enumerate(lines):
        # Skip processing if we're in a code block - preserve it exactly
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            formatted_lines.append(line)
            continue
            
        if in_code_block:
            formatted_lines.append(line)
            continue
        
        # Keep track of empty lines to avoid duplicates
        if not line.strip():
            if not previous_line_was_empty:
                formatted_lines.append(line)
                previous_line_was_empty = True
            continue
        else:
            previous_line_was_empty = False
        
        # If the line already has markdown heading, don't modify it
        if re.match(r'^#{1,3}\s+.+', line.strip()):
            # Ensure proper spacing around headings
            if i > 0 and formatted_lines and formatted_lines[-1].strip():
                formatted_lines.append("")  # Add space before heading if needed
            formatted_lines.append(line)
            if i < len(lines) - 1 and lines[i+1].strip():
                formatted_lines.append("")  # Add space after heading if needed
            continue
        
        # Check if this line looks like a heading but doesn't have markdown
        is_header = False
        header_patterns = [
            r'^(Introduction|Overview|Background|History|Analysis|Conclusion|Summary|Methods|Results|Discussion|Findings|References)(\s*:)?$',
            r'^(Chapter|Section|Part)\s+\d+[:.]\s*.+$',
            r'^([A-Z][a-z]+(\s+[A-Z][a-z]+){0,4})(\s*:)?$',  # Capitalized phrases up to 5 words
            r'^([IVX]+\.)\s+.+$',  # Roman numeral patterns
            r'^(\d+\.)\s+.+$',  # Numbered headers
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, line.strip()) and not (i > 0 and len(lines[i-1].strip()) > 100):
                # This looks like a section heading (not following a long paragraph)
                is_header = True
                # Add space before heading if needed
                if i > 0 and formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append("")
                    
                line = f"## {line.strip()}"
                formatted_lines.append(line)
                
                # Add space after heading if needed
                if i < len(lines) - 1 and lines[i+1].strip():
                    formatted_lines.append("")
                    
                break
        
        if is_header:
            continue
        
        # Check for subheadings
        subheader_patterns = [
            r'^(\d+\.\d+\.)\s+.+$',  # Numbered subheaders
            r'^([a-z]\.|\([a-z]\))\s+.+$',  # Letter subheaders
        ]
        
        for pattern in subheader_patterns:
            if re.match(pattern, line.strip()) and not (i > 0 and len(lines[i-1].strip()) > 100):
                is_header = True
                # Add space before heading if needed
                if i > 0 and formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append("")
                    
                line = f"### {line.strip()}"
                formatted_lines.append(line)
                
                # Add space after heading if needed
                if i < len(lines) - 1 and lines[i+1].strip():
                    formatted_lines.append("")
                break
        
        if is_header:
            continue
            
        # Ensure source citations are bold
        if '[Source' in line:
            line = re.sub(r'\[Source (\d+)\]', r'**[Source \1]**', line)
            
        # Format quotes if they're not already formatted
        if ('quote' in line.lower() or 'said' in line.lower() or '"' in line) and not line.startswith('>'):
            quote_match = re.search(r'"([^"]+)"', line)
            if quote_match:
                quoted_text = quote_match.group(1)
                line = line.replace(f'"{quoted_text}"', f'> "{quoted_text}"')
        
        # Enhance emphasis for important terms that aren't already bold/italicized
        # Only if they're not already formatted with markdown
        if not re.search(r'\*\*|\*|__|\b_\w+_\b', line):
            # Important concepts and terminology
            important_terms = [
                r'\b(key|critical|vital|essential|fundamental|significant|major)\s+(concept|finding|discovery|principle|factor|element)\b',
                r'\b(first|earliest|latest|most recent|revolutionary|groundbreaking|innovative)\s+(study|research|work|analysis|paper|publication)\b',
                r'\b(notable|famous|renowned|acclaimed|prominent|leading|foremost)\s+(expert|researcher|scientist|author|figure|authority)\b'
            ]
            
            for pattern in important_terms:
                line = re.sub(pattern, r'**\1 \2**', line, flags=re.IGNORECASE)
            
            # Dates and numerical data
            line = re.sub(r'\b(\d{4}(-\d{2}-\d{2})?)\b', r'**\1**', line)  # Dates
            
            # Technical terms - look for capitalized words mid-sentence
            line = re.sub(r'(?<=\s)([A-Z][a-z]+(?:[A-Z][a-z]+)+)(?=\s|\.|\,)', r'*\1*', line)
        
        formatted_lines.append(line)
    
    # Join the lines back together
    formatted_text = '\n'.join(formatted_lines)
    
    # Ensure UI clarity with titles and sections
    if not formatted_text.startswith("# "):
        title_match = re.search(r'## (.+?)(\n|$)', formatted_text)
        if title_match:
            title = title_match.group(1).strip()
            formatted_text = f"# {title}\n\n{formatted_text.replace(f'## {title}', '', 1)}"
    
    # Add end tags for any unclosed code blocks
    if formatted_text.count("```") % 2 != 0:
        formatted_text += "\n```"
    
    return formatted_text

if __name__ == "__main__":
    main()