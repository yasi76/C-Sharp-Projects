import re
import json
import asyncio
import aiohttp
import socket
import ssl
import logging
from typing import List, Optional, Dict, Tuple, Set
from collections import Counter
from tqdm import tqdm
from urllib.parse import urlparse
from functools import lru_cache
from bs4 import BeautifulSoup
import html
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('company_extractor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class ExtractionSettings:
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 15
    USER_AGENT = "Mozilla/5.0 (compatible; CompanyExtractor/1.0; +https://github.com/companyextractor)"
    ACCEPTED_LANGUAGES = "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
    RETRY_ATTEMPTS = 3
    MAX_CACHE_SIZE = 1000

# Enhanced company suffixes with regex patterns
COMPANY_SUFFIXES = [
    "GmbH", "UG", "AG", r"e\.?V\.?", "GbR", "Inc", "Ltd", "SAS", "BV", "AB",
    r"S\.?L\.?", "Oy", "KG", "SE", "LLC", "PLC", "Corp", r"Co\.", "Limited", r"S\.?A\.?",
    "NV", r"S\.?p\.?A\.?", "LP", "LLP", r"Pte\.? Ltd\.?", r"S\.?à r\.?l\.?", r"B\.?V\.?", "KGaA"
]

# Common words and phrases to ignore (expanded)
COMMON_WORDS = {
    "the", "and", "for", "with", "our", "your", "from", "this", "that", "about",
    "contact", "home", "privacy", "terms", "imprint", "legal", "cookies", "blog",
    "news", "careers", "team", "product", "products", "service", "services", "solutions",
    "impressum", "datenschutz", "agb", "kontakt", "cookie", "policy", "rights", "reserved",
    "success", "stories", "redirecting", "loading", "please", "wait", "welcome", "hello",
    "commercial", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "pilot", "clinics",
    "interested", "werden", "sie", "studienteilnehmer", "vielen", "dank", "dieser",
    "account", "ist", "zur", "zeit", "nicht", "erreichbar", "unternehmen", "responses",
    "may", "be", "limited", "ablehnen", "akzeptieren", "diese", "vertrauen", "unsere",
    "expertise", "wichtige", "funktionen", "einfache", "individuelle", "trainingsplne",
    "zuverlssige", "diagnose", "prozesse", "aller", "verschreibungen", "ab", "ulrike",
    "becker", "diabetologist", "dr", "die", "besten", "inhalte", "julian", "nast",
    "by", "close", "werden", "join", "studies", "regain", "control", "again", "fr",
    "zahnrzte", "vorname", "nachname", "email", "adresse"
}

# Enhanced blacklist patterns for content that should never be company names
BLACKLIST_PATTERNS = [
    r'^(success|stories|redirecting|loading|please|wait|welcome).*',
    r'^(commercial|pilot|clinics|interested).*',
    r'^(werden|sie|studienteilnehmer).*',
    r'^(vielen|dank).*',
    r'^(dieser|account|ist|zur|zeit).*',
    r'^(responses|may|be|limited).*',
    r'^(ablehnen|akzeptieren|diese).*',
    r'^(vertrauen|unsere|expertise).*',
    r'^(wichtige|funktionen|einfache).*',
    r'^(individuelle|trainingsplne).*',
    r'^(zuverlssige|diagnose).*',
    r'^(aller|verschreibungen|ab).*',
    r'^(die|besten|inhalte|ab).*',
    r'^(julian|nast).*',
    r'^(by|close).*',
    r'^(regain|control|again).*',
    r'^(vorname|nachname|email|adresse).*',
    r'.*\d{4}.*',  # Years
    r'^(dr|prof)\.?\s+.*',  # Doctor titles
    r'.*@.*',  # Email addresses
    r'^(home|about|contact|impressum)$',
    r'^(cookie|privacy|terms|legal)$'
]

# Priority selectors for finding company names
PRIORITY_SELECTORS = [
    # Schema.org markup
    'script[type="application/ld+json"]',
    # Meta tags
    'meta[property="og:site_name"]',
    'meta[name="application-name"]',
    'meta[property="og:title"]',
    # Header elements
    'h1',
    'header h1, header h2, header h3',
    '.header h1, .header h2, .header h3',
    # Logo areas
    '.logo', '#logo', '[class*="logo"]',
    '.brand', '#brand', '[class*="brand"]',
    # Navigation
    '.navbar-brand', '.nav-brand',
    # Footer copyright
    'footer', '.footer',
    # Title tags
    'title'
]

# HTML meta tags that often contain company names
META_TAGS = [
    'og:site_name', 'og:title', 'twitter:site', 'application-name',
    'apple-mobile-web-app-title', 'company', 'organization', 'author'
]

def clean_text(text: str) -> str:
    """Clean and normalize text with better HTML and encoding handling."""
    if not text:
        return ""

    # Handle encoding issues
    try:
        text = text.encode('ascii', 'ignore').decode('utf-8')
    except:
        pass

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags and comments
    soup = BeautifulSoup(text, 'html.parser')
    for element in soup(['script', 'style', 'noscript', 'meta', 'link', 'comment']):
        element.decompose()

    # Get text with proper spacing
    text = ' '.join(soup.stripped_strings)

    # Remove invisible characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Normalize whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\u00a0\u1680\u2000-\u200f\u2028-\u202f\u205f\u3000\ufeff]', ' ', text)

    return text.strip()

def extract_from_domain(url: str) -> Optional[str]:
    """Enhanced domain-based company name extraction."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.lower().replace('www.', '').split('/')[0].split(':')[0]
        
        # Remove TLD
        domain_parts = domain.split('.')
        if len(domain_parts) >= 2:
            company_part = domain_parts[0]
            
            # Handle common patterns
            if company_part in ['shop', 'app', 'api', 'portal', 'web']:
                if len(domain_parts) >= 3:
                    company_part = domain_parts[1]
            
            # Clean and capitalize
            company_part = re.sub(r'[^a-zA-Z0-9]', '', company_part)
            if len(company_part) >= 3:
                return company_part.capitalize()
    except:
        pass
    return None

async def is_website_reachable(url: str) -> bool:
    """Check if a website is reachable without loading the full page."""
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc or parsed.path.split('/')[0]

        # First try DNS resolution
        loop = asyncio.get_running_loop()
        await loop.getaddrinfo(hostname, None)

        # Then try SSL handshake
        context = ssl.create_default_context()
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(hostname, 443, ssl=context),
            timeout=5
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception as e:
        logger.debug(f"Website {url} not reachable: {str(e)}")
        return False

def extract_from_meta(soup: BeautifulSoup) -> List[str]:
    """Extract potential company names from meta tags."""
    candidates = []

    for tag in META_TAGS:
        meta = soup.find('meta', {'property': tag}) or soup.find('meta', {'name': tag})
        if meta and meta.get('content'):
            content = meta['content'].strip()
            if len(content) > 3 and not is_blacklisted(content):
                candidates.append(content)

    title = soup.title.string if soup.title else None
    if title and len(title) > 3 and not is_blacklisted(title):
        candidates.append(title.strip())

    return candidates

def extract_from_schema(soup: BeautifulSoup) -> List[str]:
    """Extract company names from schema.org markup."""
    candidates = []
    for script in soup.find_all('script', {'type': 'application/ld+json'}):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                # Handle different schema types
                if data.get('@type') in ['Organization', 'Corporation', 'LocalBusiness', 'Company']:
                    if data.get('name'):
                        candidates.append(data['name'])
                    if data.get('legalName'):
                        candidates.append(data['legalName'])
                elif data.get('publisher', {}).get('name'):
                    candidates.append(data['publisher']['name'])
                elif data.get('author', {}).get('name'):
                    author_name = data['author']['name']
                    # Only add if it looks like a company
                    if any(suffix in author_name for suffix in ['GmbH', 'AG', 'UG', 'Inc', 'Ltd']):
                        candidates.append(author_name)
        except (json.JSONDecodeError, AttributeError):
            continue
    return candidates

def is_blacklisted(text: str) -> bool:
    """Check if text matches blacklist patterns."""
    text_lower = text.lower().strip()
    
    # Check against common words
    if any(word in text_lower for word in COMMON_WORDS):
        return True
    
    # Check against blacklist patterns
    for pattern in BLACKLIST_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False

def score_company_name(name: str, url: str = "") -> int:
    """Enhanced scoring for potential company names."""
    if not name or is_blacklisted(name):
        return -10
    
    score = 0
    name_lower = name.lower()

    # Length score (prefer medium-length names)
    length = len(name)
    if 5 <= length <= 50:
        score += 3
    elif 3 <= length < 5:
        score += 1
    elif length > 50:
        score -= 2

    # Legal form presence (strong indicator)
    for suffix in COMPANY_SUFFIXES:
        if re.search(rf'\b{suffix}\b', name, flags=re.IGNORECASE):
            score += 5
            break

    # Capitalization pattern (proper case)
    if re.match(r'^[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+)*', name):
        score += 2

    # Domain match bonus
    if url:
        domain_part = extract_from_domain(url)
        if domain_part and domain_part.lower() in name_lower:
            score += 3

    # Penalty for generic terms
    generic_terms = ['website', 'homepage', 'portal', 'platform', 'system', 'app', 'application']
    if any(term in name_lower for term in generic_terms):
        score -= 2

    # Bonus for business-related terms
    business_terms = ['technologies', 'solutions', 'systems', 'software', 'health', 'medical', 'ai']
    if any(term in name_lower for term in business_terms):
        score += 1

    # Penalty for numbers in inappropriate places
    if re.search(r'^\d+', name) or re.search(r'\d{4}', name):
        score -= 2

    return score

def extract_from_priority_selectors(soup: BeautifulSoup) -> List[str]:
    """Extract company names using priority CSS selectors."""
    candidates = []
    
    for selector in PRIORITY_SELECTORS:
        try:
            elements = soup.select(selector)
            for element in elements[:3]:  # Limit to first 3 matches per selector
                if selector == 'script[type="application/ld+json"]':
                    # Handle JSON-LD separately
                    continue
                
                text = element.get_text(strip=True) if hasattr(element, 'get_text') else str(element)
                if text and len(text) >= 3 and not is_blacklisted(text):
                    # Clean up common prefixes/suffixes
                    text = re.sub(r'^(by\s+|©\s*|copyright\s*)', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'\s*(all rights reserved|©).*$', '', text, flags=re.IGNORECASE)
                    text = text.strip()
                    
                    if text:
                        candidates.append(text)
        except Exception as e:
            logger.debug(f"Error with selector {selector}: {e}")
            continue
    
    return candidates

def extract_company_names_from_text(text: str) -> List[str]:
    """Extract company names from text using improved regex patterns."""
    if not text:
        return []
    
    candidates = []
    
    # Pattern for companies with legal suffixes
    pattern1 = rf'\b([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+)*)\s+(?:{'|'.join(COMPANY_SUFFIXES)})\b'
    matches1 = re.findall(pattern1, text, flags=re.IGNORECASE)
    candidates.extend(matches1)
    
    # Pattern for multi-word capitalized names (likely companies)
    pattern2 = r'\b([A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+){1,3})\b'
    matches2 = re.findall(pattern2, text)
    for match in matches2:
        if not is_blacklisted(match) and len(match.split()) <= 4:
            candidates.append(match)
    
    return candidates

def extract_from_copyright(soup: BeautifulSoup) -> List[str]:
    """Extract company names from copyright notices."""
    candidates = []
    
    # Look for copyright text
    copyright_patterns = [
        r'©\s*\d{4}[^a-zA-Z]*([^.]{3,30})',
        r'copyright\s*\d{4}[^a-zA-Z]*([^.]{3,30})',
        r'©\s*([^.\d]{3,30})\s*\d{4}',
        r'copyright\s*([^.\d]{3,30})\s*\d{4}'
    ]
    
    footer_text = ""
    footer = soup.find('footer') or soup.find(class_=re.compile('footer', re.IGNORECASE))
    if footer:
        footer_text = footer.get_text()
    
    # Also check entire page for copyright
    page_text = soup.get_text()
    
    for text in [footer_text, page_text]:
        for pattern in copyright_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            for match in matches:
                cleaned = match.strip()
                if not is_blacklisted(cleaned) and len(cleaned) >= 3:
                    candidates.append(cleaned)
    
    return candidates

@retry(stop=stop_after_attempt(ExtractionSettings.RETRY_ATTEMPTS),
       wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_url(session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
    """Fetch URL content with retry logic."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=ExtractionSettings.REQUEST_TIMEOUT)) as response:
            if response.status >= 400:
                raise aiohttp.ClientError(f"HTTP status {response.status}")
            return (url, await response.text())
    except Exception as e:
        logger.warning(f"Error fetching {url}: {str(e)}")
        raise

async def extract_from_url(url: str, session: aiohttp.ClientSession) -> dict:
    """Enhanced URL processing with improved extraction strategies."""
    try:
        original_url = url
        url = normalize_url(url)

        # First try domain extraction as baseline
        domain_company = extract_from_domain(url)

        # Check if website is reachable
        if not await is_website_reachable(url):
            if domain_company:
                return {
                    "url": original_url,
                    "company": domain_company,
                    "candidates": [domain_company],
                    "status": "domain_fallback",
                    "confidence": "low"
                }
            return {"url": original_url, "company": None, "status": "unreachable", "error": "Website not reachable"}

        # Fetch the URL content
        try:
            _, content = await fetch_url(session, url)
        except Exception as e:
            if domain_company:
                return {
                    "url": original_url,
                    "company": domain_company,
                    "candidates": [domain_company],
                    "status": "domain_fallback",
                    "confidence": "low"
                }
            return {"url": original_url, "company": None, "status": "error", "error": str(e)}

        soup = BeautifulSoup(content, 'html.parser')
        
        # Collect candidates from multiple sources
        all_candidates = []

        # 1. Schema.org markup (highest priority)
        schema_candidates = extract_from_schema(soup)
        all_candidates.extend([(c, 10) for c in schema_candidates])

        # 2. Meta tags (high priority)
        meta_candidates = extract_from_meta(soup)
        all_candidates.extend([(c, 8) for c in meta_candidates])

        # 3. Priority selectors (medium-high priority)
        selector_candidates = extract_from_priority_selectors(soup)
        all_candidates.extend([(c, 6) for c in selector_candidates])

        # 4. Copyright notices (medium priority)
        copyright_candidates = extract_from_copyright(soup)
        all_candidates.extend([(c, 5) for c in copyright_candidates])

        # 5. Text pattern matching (lower priority)
        text_content = clean_text(content)
        text_candidates = extract_company_names_from_text(text_content)
        all_candidates.extend([(c, 3) for c in text_candidates])

        # 6. Domain fallback (lowest priority)
        if domain_company:
            all_candidates.append((domain_company, 1))

        # Score and rank candidates
        scored_candidates = []
        seen = set()
        
        for candidate, base_score in all_candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                content_score = score_company_name(candidate, url)
                total_score = content_score + base_score
                
                if total_score > 0:
                    scored_candidates.append((total_score, candidate, base_score))

        if scored_candidates:
            # Sort by total score
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            best_score, best_candidate, source_score = scored_candidates[0]
            
            # Post-process the best candidate
            best_candidate = post_process_company_name(best_candidate)
            
            if best_candidate and len(best_candidate) >= 3:
                confidence = "high" if best_score >= 10 else "medium" if best_score >= 5 else "low"
                
                return {
                    "url": original_url,
                    "company": best_candidate,
                    "candidates": [c[1] for c in scored_candidates[:5]],  # Top 5 candidates
                    "status": "success",
                    "confidence": confidence,
                    "score": best_score
                }

        # Final fallback to domain
        if domain_company:
            return {
                "url": original_url,
                "company": domain_company,
                "candidates": [domain_company],
                "status": "domain_fallback",
                "confidence": "low"
            }

        return {"url": original_url, "company": None, "status": "not_found"}

    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}", exc_info=True)
        domain_company = extract_from_domain(url)
        if domain_company:
            return {
                "url": original_url,
                "company": domain_company,
                "candidates": [domain_company],
                "status": "domain_fallback",
                "confidence": "low"
            }
        return {"url": original_url, "company": None, "status": "error", "error": str(e)}

def post_process_company_name(name: str) -> str:
    """Post-process and clean up extracted company name."""
    if not name:
        return ""
    
    # Remove leading/trailing non-alphanumeric characters
    name = re.sub(r'^[^a-zA-ZÀ-ÖØ-ßà-öø-ÿ0-9]*', '', name)
    name = re.sub(r'[^a-zA-ZÀ-ÖØ-ßà-öø-ÿ0-9\s\-&.]*$', '', name)
    
    # Clean up common prefixes
    name = re.sub(r'^(by\s+|©\s*|copyright\s*)', '', name, flags=re.IGNORECASE)
    
    # Clean up common suffixes
    name = re.sub(r'\s*(all rights reserved|©|\.com|\.de|\.org).*$', '', name, flags=re.IGNORECASE)
    
    # Remove blacklisted words
    words = name.split()
    filtered_words = []
    for word in words:
        if word.lower() not in COMMON_WORDS:
            filtered_words.append(word)
    
    name = ' '.join(filtered_words)
    
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def normalize_url(url: str) -> str:
    """Normalize URLs to consistent format."""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Remove fragments and query params for company name extraction
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

@lru_cache(maxsize=ExtractionSettings.MAX_CACHE_SIZE)
def get_domain(url: str) -> str:
    """Extract domain from URL with caching for performance."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        return domain.lower().replace('www.', '').split('/')[0].split(':')[0]
    except:
        return ""

async def process_urls(urls: List[str]) -> List[dict]:
    """Process multiple URLs asynchronously with progress tracking."""
    results = []
    connector = aiohttp.TCPConnector(limit=ExtractionSettings.MAX_CONCURRENT_REQUESTS)

    headers = {
        "User-Agent": ExtractionSettings.USER_AGENT,
        "Accept-Language": ExtractionSettings.ACCEPTED_LANGUAGES,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [extract_from_url(url, session) for url in urls]

        with tqdm(total=len(urls), desc="Processing URLs", unit="URL") as pbar:
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    results.append(result)

                    if result['status'] not in ('success', 'domain_fallback'):
                        logger.info(f"Failed to extract from {result['url']}: {result['status']}")
                except Exception as e:
                    logger.error(f"Exception processing URL: {e}", exc_info=True)
                    results.append({"url": "unknown", "company": None, "status": "error", "error": str(e)})
                finally:
                    pbar.update(1)

    return results

def analyze_results(results: List[dict]) -> Dict[str, int]:
    """Analyze and summarize extraction results."""
    analysis = {
        'total': len(results),
        'success': 0,
        'domain_fallback': 0,
        'not_found': 0,
        'unreachable': 0,
        'error': 0,
        'high_confidence': 0,
        'medium_confidence': 0,
        'low_confidence': 0
    }

    for result in results:
        analysis[result['status']] += 1
        if 'confidence' in result:
            analysis[f"{result['confidence']}_confidence"] += 1

    return analysis

def print_results(results: List[dict]):
    """Print formatted results with confidence levels."""
    analysis = analyze_results(results)

    print("\n=== Extraction Results ===")
    print(f"Total URLs processed: {analysis['total']}")
    print(f"Successfully extracted: {analysis['success']}")
    print(f"Domain fallback: {analysis['domain_fallback']}")
    print(f"Not found: {analysis['not_found']}")
    print(f"Unreachable URLs: {analysis['unreachable']}")
    print(f"Errors: {analysis['error']}")
    print(f"High confidence: {analysis['high_confidence']}")
    print(f"Medium confidence: {analysis['medium_confidence']}")
    print(f"Low confidence: {analysis['low_confidence']}\n")

    print("=== Detailed Results ===")
    for result in results:
        status = result['status'].upper()
        confidence = result.get('confidence', '').upper()
        score = result.get('score', '')
        
        if result['company']:
            conf_str = f" ({confidence})" if confidence else ""
            score_str = f" [Score: {score}]" if score else ""
            print(f"{status.ljust(15)} {result['url']} -> {result['company']}{conf_str}{score_str}")
        else:
            print(f"{status.ljust(15)} {result['url']} -> NOT FOUND ({result.get('error', '')})")

async def main():
    test_urls = [
        'https://www.acalta.de',
        'https://www.actimi.com',
        'https://www.emmora.de',
        'https://www.alfa-ai.com',
        'https://www.apheris.com',
        'https://www.aporize.com/',
        'https://www.arztlena.com/',
        'https://shop.getnutrio.com/',
        'https://www.auta.health/',
        'https://visioncheckout.com/',
        'https://www.avayl.tech/',
        'https://www.avimedical.com/avi-impact',
        'https://de.becureglobal.com/',
        'https://bellehealth.co/de/',
        'https://www.biotx.ai/',
        'https://www.brainjo.de/',
        'https://brea.app/',
        'https://breathment.com/',
        'https://de.caona.eu/',
        'https://www.careanimations.de/',
        'https://sfs-healthcare.com',
        'https://www.climedo.de/',
        'https://www.cliniserve.de/',
        'https://cogthera.de/#erfahren',
        'https://www.comuny.de/',
        'https://curecurve.de/elina-app/',
        'https://www.cynteract.com/de/rehabilitation',
        'https://www.healthmeapp.de/de/',
        'https://deepeye.ai/',
        'https://www.deepmentation.ai/',
        'https://denton-systems.de/',
        'https://www.derma2go.com/',
        'https://www.dianovi.com/',
        'http://dopavision.com/',
        'https://www.dpv-analytics.com/',
        'http://www.ecovery.de/',
        'https://elixionmedical.com/',
        'https://www.empident.de/',
        'https://eye2you.ai/',
        'https://www.fitwhit.de',
        'https://www.floy.com/',
        'https://fyzo.de/assistant/',
        'https://www.gesund.de/app',
        'https://www.glaice.de/',
        'https://gleea.de/',
        'https://www.guidecare.de/',
        'https://www.apodienste.com/',
        'https://www.help-app.de/',
        'https://www.heynanny.com/',
        'https://incontalert.de/',
        'https://home.informme.info/',
        'https://www.kranushealth.com/de/therapien/haeufiger-harndrang',
        'https://www.kranushealth.com/de/therapien/inkontinenz'
    ]

    # Clean URLs (some had incorrect protocols)
    cleaned_urls = []
    for url in test_urls:
        if url.startswith('https:/') and not url.startswith('https://'):
            url = url.replace('https:/', 'https://')
        elif url.startswith('http:/') and not url.startswith('http://'):
            url = url.replace('http:/', 'http://')
        cleaned_urls.append(url)

    results = await process_urls(cleaned_urls)
    print_results(results)
    
    # Calculate accuracy against ground truth
    ground_truth = [
        'Acalta GmbH', 'Actimi GmbH', 'Actimi GmbH', 'ALFA AI GmbH', 'apheris AI GmbH',
        'Aporize', 'Artificy GmbH', 'Aurora Life Sciene GmbH', 'Auta Health UG',
        'auvisus GmbH', 'AVAYL GmbH', 'Avi Medical Operations GmbH', 'BECURE GmbH',
        'Belle Health GmbH', 'biotx.ai GmbH', 'brainjo GmbH', 'Brea Health GmbH',
        'Breathment GmbH', 'Caona Health GmbH', 'CAREANIMATIONS GmbH', 'Change IT Solutions GmbH',
        'Climedo Health GmbH', 'Clinicserve GmbH', 'Cogthera GmbH', 'comuny GmbH',
        'CureCurve Medical AI GmbH', 'Cynteract GmbH', 'Declareme GmbH', 'deepeye medical GmbH',
        'deepmentation UG', 'Denton Systems GmbH', 'derma2go Deutschland GmbH',
        'dianovi GmbH (ehem. MySympto)', 'Dopavision GmbH', 'dpv-analytics GmbH',
        'eCovery GmbH', 'Elixion Medical', 'Empident GmbH', 'eye2you',
        'FitwHit & LABOR FÜR BIOMECHANIK der JLU-Gießen', 'Floy GmbH', 'fyzo GmbH',
        'gesund.de GmbH & Co. KG', 'GLACIE Health UG', 'Gleea Educational Software GmbH',
        'GuideCare GmbH', 'Healthy Codes GmbH', 'Help Mee Schmerztherapie GmbH',
        'heynannyly GmbH', 'inContAlert GmbH', 'InformMe GmbH', 'Kranus Health GmbH',
        'Kranus Health GmbH'
    ]
    
    print(f"\n=== Accuracy Analysis ===")
    correct_extractions = 0
    
    for i, result in enumerate(results):
        if i < len(ground_truth) and result.get('company'):
            extracted = result['company'].lower()
            expected = ground_truth[i].lower()
            
            # Fuzzy matching - check if main company name is present
            expected_clean = re.sub(r'\s+(gmbh|ag|ug|inc|ltd).*$', '', expected).strip()
            if expected_clean in extracted or any(word in extracted for word in expected_clean.split() if len(word) > 3):
                correct_extractions += 1
                
    accuracy = (correct_extractions / len(ground_truth)) * 100 if ground_truth else 0
    print(f"Accuracy: {accuracy:.1f}% ({correct_extractions}/{len(ground_truth)})")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())