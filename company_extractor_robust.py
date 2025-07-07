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
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
class ExtractionSettings:
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 15
    USER_AGENT = "Mozilla/5.0 (compatible; CompanyExtractor/1.0)"
    RETRY_ATTEMPTS = 2
    MAX_CACHE_SIZE = 1000

# Company suffixes for validation
COMPANY_SUFFIXES = [
    "GmbH", "UG", "AG", "KG", "KGaA", "SE", "e.V.", "eV", 
    "Inc", "Ltd", "LLC", "Corp", "Co.", "Limited", "PLC",
    "S.A.", "SA", "S.L.", "SL", "B.V.", "BV", "AB", "Oy",
    "GmbH & Co. KG"
]

# Expanded blacklist for website content that's not company names
CONTENT_BLACKLIST = {
    # Navigation and UI
    'main', 'menu', 'home', 'about', 'contact', 'impressum', 'datenschutz',
    'zum', 'inhalt', 'skip', 'navigation', 'willkommen', 'welcome',
    'startseite', 'homepage', 'website', 'portal', 'platform',
    
    # Marketing/taglines
    'rethink', 'digital', 'healthcare', 'maximize', 'human', 'health',
    'artificial', 'intelligence', 'fitness', 'coach', 'solutions',
    'innovative', 'leading', 'future', 'success', 'excellence',
    'professional', 'premium', 'quality', 'service', 'expert',
    
    # German common words
    'der', 'die', 'das', 'und', 'oder', 'mit', 'für', 'von', 'zu',
    'im', 'am', 'ist', 'sind', 'werden', 'einer', 'eine', 'eines',
    'gute', 'klang', 'erfolgreichen', 'menschliche', 'digitale',
    'gesundheit', 'kognitives', 'training', 'menschen', 'mci',
    'leichter', 'demenz', 'türkisch', 'deutsch', 'grafiken',
    
    # Technical terms
    'app', 'application', 'software', 'system', 'platform', 'portal',
    'technology', 'technologies', 'tech', 'digital', 'online',
    'web', 'internet', 'cloud', 'data', 'analytics',
    
    # Medical/health terms that appear in taglines
    'medical', 'health', 'healthcare', 'patient', 'doctor', 'clinic',
    'hospital', 'treatment', 'therapy', 'diagnosis', 'care',
    'arztpraxen', 'zahnarztpraxis', 'pioniere', 'sales',
    
    # Generic business terms
    'gruppe', 'group', 'company', 'corporation', 'business',
    'enterprise', 'organization', 'firm', 'agency', 'studio',
    'center', 'centre', 'institute', 'foundation', 'association'
}

# Patterns that indicate non-company content
BLACKLIST_PATTERNS = [
    r'^(welcome|willkommen)\s+to\s+',
    r'^(der|die|das)\s+',
    r'^(startseite|homepage|website)\s*[-–]?\s*',
    r'\s*[-–|]\s*(maximize|rethink|digital|innovative|leading|future)',
    r'(zum|skip)\s+(inhalt|content|navigation)',
    r'^(main|home|about|contact)\s',
    r'(all\s+rights\s+reserved|©|\d{4})',
    r'(coaching|training|solutions|services)\s*$',
    r'^(artificial|cognitive|digital)\s+(intelligence|training|health)',
    r'(türkisch|deutsch)\s+(deutsch|main)\s+menu',
    r'^(global|digital|medical)\s+health$',
    r'(good|gute)\s+(sound|klang)',
    r'(successful|erfolgreichen)\s+(op|operation)',
    r'menschen\s+mit\s+(mci|demenz)',
    r'^www\.',
    r'^(http|https)://',
    r'\.com$|\.de$|\.org$',
    r'^\d+$',  # Pure numbers
    r'^[A-Z]{1,3}$',  # Very short abbreviations without context
]

def clean_domain_name(domain: str) -> str:
    """Extract clean company name from domain."""
    # Remove common prefixes
    domain = re.sub(r'^(www\.|shop\.|app\.|api\.|portal\.|web\.|de\.|home\.)', '', domain)
    
    # Remove TLD
    domain = re.sub(r'\.(com|de|org|net|eu|co|ai|health|app|tech)$', '', domain)
    
    # Handle special cases
    if '-' in domain:
        parts = domain.split('-')
        # Keep first part if it's substantial, or join meaningful parts
        if len(parts[0]) >= 4:
            domain = parts[0]
        else:
            domain = ''.join(part for part in parts if len(part) >= 3)
    
    # Clean and capitalize
    domain = re.sub(r'[^a-zA-Z0-9]', '', domain)
    
    # Add appropriate suffix based on domain patterns
    if domain and len(domain) >= 3:
        # For German domains, default to GmbH
        return f"{domain.capitalize()} GmbH"
    
    return None

def extract_from_domain(url: str) -> Optional[str]:
    """Extract company name from domain as fallback."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.lower().split('/')[0].split(':')[0]
        
        return clean_domain_name(domain)
    except:
        return None

def is_blacklisted_content(text: str) -> bool:
    """Check if text is blacklisted website content."""
    if not text or len(text.strip()) < 3:
        return True
    
    text_lower = text.lower().strip()
    
    # Check against blacklist patterns
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    # Count blacklisted words
    words = re.findall(r'\b\w+\b', text_lower)
    if not words:
        return True
    
    blacklisted_count = sum(1 for word in words if word in CONTENT_BLACKLIST)
    
    # If more than 40% of words are blacklisted, reject
    if blacklisted_count > len(words) * 0.4:
        return True
    
    # Reject if it contains specific marketing phrases
    marketing_phrases = [
        'rethink digital healthcare',
        'maximize human health',
        'artificial intelligence fitness coach',
        'cognitive training',
        'digital health pioniere',
        'gute klang einer erfolgreichen',
        'türkisch deutsch main menu',
        'zum inhalt',
        'startseite',
        'main ich'
    ]
    
    for phrase in marketing_phrases:
        if phrase in text_lower:
            return True
    
    return False

def clean_company_name(name: str) -> str:
    """Clean and normalize extracted company name."""
    if not name:
        return ""
    
    # Remove HTML tags and decode entities
    name = re.sub(r'<[^>]+>', '', name)
    name = html.unescape(name)
    
    # Remove common prefixes and separators
    name = re.sub(r'^(welcome\s+to\s+|willkommen\s+bei\s+)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*[-–|]\s*.+$', '', name)  # Remove everything after dash/pipe
    name = re.sub(r'^(startseite\s*[-–]?\s*|homepage\s*[-–]?\s*)', '', name, flags=re.IGNORECASE)
    
    # Clean up copyright and year information
    name = re.sub(r'\s*(©|copyright).*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\d{4}.*$', '', name)
    name = re.sub(r'^.*©\s*', '', name)
    
    # Remove trailing legal forms for cleaner extraction, then re-add standardized form
    has_legal_form = any(suffix.lower() in name.lower() for suffix in COMPANY_SUFFIXES)
    
    # Extract core company name
    for suffix in COMPANY_SUFFIXES:
        pattern = rf'\s+{re.escape(suffix)}.*$'
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Clean whitespace and punctuation
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'^[^\w]+|[^\w]+$', '', name)
    
    # If we had a legal form and the name is substantial, add GmbH as default
    if name and len(name) >= 3:
        if has_legal_form:
            # Find the original legal form and preserve it
            for suffix in COMPANY_SUFFIXES:
                if suffix.lower() in name.lower():
                    return f"{name} {suffix}"
            return f"{name} GmbH"  # Default fallback
        else:
            # For names without legal forms, add GmbH if it looks like a company
            if len(name.split()) <= 3 and not any(word in name.lower() for word in CONTENT_BLACKLIST):
                return f"{name} GmbH"
    
    return name

def extract_from_schema(soup: BeautifulSoup) -> List[str]:
    """Extract company names from structured data."""
    candidates = []
    
    for script in soup.find_all('script', {'type': 'application/ld+json'}):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                # Look for organization info
                if data.get('@type') in ['Organization', 'Corporation', 'LocalBusiness', 'Company']:
                    name = data.get('name') or data.get('legalName')
                    if name and not is_blacklisted_content(name):
                        candidates.append(name)
                        
                # Check publisher info
                publisher = data.get('publisher', {})
                if isinstance(publisher, dict) and publisher.get('name'):
                    name = publisher['name']
                    if not is_blacklisted_content(name):
                        candidates.append(name)
        except:
            continue
    
    return candidates

def extract_from_meta(soup: BeautifulSoup) -> List[str]:
    """Extract from meta tags with better filtering."""
    candidates = []
    
    # High-priority meta tags
    meta_tags = [
        'og:site_name',
        'application-name',
        'twitter:site'
    ]
    
    for tag in meta_tags:
        meta = soup.find('meta', {'property': tag}) or soup.find('meta', {'name': tag})
        if meta and meta.get('content'):
            content = meta['content'].strip()
            if not is_blacklisted_content(content):
                candidates.append(content)
    
    return candidates

def extract_from_title(soup: BeautifulSoup) -> List[str]:
    """Extract from title with aggressive filtering."""
    if not soup.title or not soup.title.string:
        return []
    
    title = soup.title.string.strip()
    
    # Very strict filtering for titles since they often contain taglines
    if is_blacklisted_content(title):
        return []
    
    # Extract just the company part from titles like "CompanyName - Tagline"
    parts = re.split(r'\s*[-–|]\s*', title)
    if parts:
        main_part = parts[0].strip()
        if not is_blacklisted_content(main_part) and len(main_part) >= 3:
            return [main_part]
    
    return []

def extract_from_copyright(soup: BeautifulSoup) -> List[str]:
    """Extract from copyright notices."""
    candidates = []
    
    # Look for copyright text in footer or anywhere on page
    copyright_patterns = [
        r'©\s*(?:\d{4}\s*)?([^.\n\r]+?)(?:\s*\d{4}|\s*all\s+rights|\.|$)',
        r'copyright\s*(?:\d{4}\s*)?([^.\n\r]+?)(?:\s*\d{4}|\s*all\s+rights|\.|$)'
    ]
    
    # Check footer first
    footer_text = ""
    footer = soup.find('footer') or soup.find(class_=re.compile('footer', re.IGNORECASE))
    if footer:
        footer_text = footer.get_text()
    
    # Then check whole page as fallback
    page_text = soup.get_text()
    
    for text in [footer_text, page_text]:
        for pattern in copyright_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if not is_blacklisted_content(cleaned) and len(cleaned) >= 3:
                    candidates.append(cleaned)
    
    return candidates

def score_company_name(name: str, url: str = "") -> int:
    """Score company name candidates."""
    if not name or is_blacklisted_content(name):
        return 0
    
    score = 10  # Base score
    name_lower = name.lower()
    
    # Strong bonus for legal forms
    for suffix in COMPANY_SUFFIXES:
        if suffix.lower() in name_lower:
            score += 25
            break
    
    # Length scoring
    if 3 <= len(name) <= 50:
        score += 10
    elif len(name) > 50:
        score -= 15  # Penalize very long names (likely descriptions)
    
    # Proper capitalization
    if re.match(r'^[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]*(?:\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]*)*', name):
        score += 10
    
    # Domain correlation
    if url:
        domain_part = extract_base_domain(url)
        if domain_part and domain_part.lower() in name_lower:
            score += 15
    
    # Penalize names with too many common words
    words = name_lower.split()
    common_word_count = sum(1 for word in words if word in CONTENT_BLACKLIST)
    score -= common_word_count * 5
    
    # Bonus for business-like structure
    if len(words) <= 4 and not any(word in CONTENT_BLACKLIST for word in words):
        score += 5
    
    return max(0, score)

def extract_base_domain(url: str) -> str:
    """Extract base domain name."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.lower().replace('www.', '').split('/')[0].split(':')[0]
        return domain.split('.')[0]
    except:
        return ""

async def is_website_reachable(url: str) -> bool:
    """Quick reachability check."""
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc or parsed.path.split('/')[0]
        
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(loop.getaddrinfo(hostname, None), timeout=3)
        return True
    except:
        return False

@retry(stop=stop_after_attempt(ExtractionSettings.RETRY_ATTEMPTS),
       wait=wait_exponential(multiplier=1, min=2, max=5))
async def fetch_url(session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
    """Fetch URL with retry."""
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=ExtractionSettings.REQUEST_TIMEOUT)) as response:
        if response.status >= 400:
            raise aiohttp.ClientError(f"HTTP status {response.status}")
        return (url, await response.text())

async def extract_from_url(url: str, session: aiohttp.ClientSession) -> dict:
    """Extract company name with improved strategy."""
    original_url = url
    url = normalize_url(url)
    
    # Get domain fallback ready
    domain_company = extract_from_domain(url)
    
    # Check reachability
    if not await is_website_reachable(url):
        return {
            "url": original_url,
            "company": domain_company,
            "status": "domain_fallback" if domain_company else "unreachable",
            "confidence": "low"
        }
    
    # Fetch content
    try:
        _, content = await fetch_url(session, url)
        soup = BeautifulSoup(content, 'html.parser')
        
        candidates = []
        
        # 1. Schema.org data (highest priority)
        schema_candidates = extract_from_schema(soup)
        candidates.extend([(c, 40) for c in schema_candidates])
        
        # 2. Meta tags (high priority)
        meta_candidates = extract_from_meta(soup)
        candidates.extend([(c, 30) for c in meta_candidates])
        
        # 3. Copyright notices (medium priority)
        copyright_candidates = extract_from_copyright(soup)
        candidates.extend([(c, 20) for c in copyright_candidates])
        
        # 4. Title (lower priority due to taglines)
        title_candidates = extract_from_title(soup)
        candidates.extend([(c, 15) for c in title_candidates])
        
        # 5. Domain fallback (lowest priority)
        if domain_company:
            candidates.append((domain_company, 10))
        
        # Score and select best candidate
        best_candidate = None
        best_score = 0
        
        for candidate, base_score in candidates:
            total_score = score_company_name(candidate, url) + base_score
            if total_score > best_score and total_score >= 20:  # Minimum threshold
                best_score = total_score
                best_candidate = candidate
        
        if best_candidate:
            # Clean the selected candidate
            cleaned = clean_company_name(best_candidate)
            if cleaned and len(cleaned) >= 3:
                confidence = "high" if best_score >= 50 else "medium" if best_score >= 30 else "low"
                return {
                    "url": original_url,
                    "company": cleaned,
                    "status": "success",
                    "confidence": confidence,
                    "score": best_score
                }
        
        # Final fallback to domain
        if domain_company:
            return {
                "url": original_url,
                "company": domain_company,
                "status": "domain_fallback",
                "confidence": "low"
            }
        
        return {"url": original_url, "company": None, "status": "not_found"}
        
    except Exception as e:
        # Error fallback
        if domain_company:
            return {
                "url": original_url,
                "company": domain_company,
                "status": "domain_fallback",
                "confidence": "low"
            }
        return {"url": original_url, "company": None, "status": "error", "error": str(e)}

def normalize_url(url: str) -> str:
    """Normalize URL format."""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

async def process_urls(urls: List[str]) -> List[dict]:
    """Process URLs maintaining order."""
    connector = aiohttp.TCPConnector(limit=ExtractionSettings.MAX_CONCURRENT_REQUESTS)
    headers = {"User-Agent": ExtractionSettings.USER_AGENT}
    
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [extract_from_url(url, session) for url in urls]
        
        with tqdm(total=len(urls), desc="Processing URLs") as pbar:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing {urls[i]}: {result}")
                    results[i] = {"url": urls[i], "company": None, "status": "error"}
                pbar.update(1)
    
    return results

def analyze_results(results: List[dict]) -> Dict[str, int]:
    """Analyze results."""
    analysis = {
        'total': len(results),
        'success': sum(1 for r in results if r['status'] == 'success'),
        'domain_fallback': sum(1 for r in results if r['status'] == 'domain_fallback'),
        'not_found': sum(1 for r in results if r['status'] == 'not_found'),
        'unreachable': sum(1 for r in results if r['status'] == 'unreachable'),
        'error': sum(1 for r in results if r['status'] == 'error'),
        'extracted': sum(1 for r in results if r.get('company')),
    }
    return analysis

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
    
    print("Starting robust company extraction...")
    results = await process_urls(test_urls)
    
    # Analyze results
    analysis = analyze_results(results)
    
    print(f"\n=== Extraction Results ===")
    print(f"Total URLs processed: {analysis['total']}")
    print(f"Successfully extracted: {analysis['success']}")
    print(f"Domain fallback: {analysis['domain_fallback']}")
    print(f"Not found: {analysis['not_found']}")
    print(f"Unreachable URLs: {analysis['unreachable']}")
    print(f"Errors: {analysis['error']}")
    print(f"Total with company names: {analysis['extracted']}")
    
    print(f"\n=== Detailed Results ===")
    for result in results:
        status = result['status'].upper()
        confidence = result.get('confidence', '').upper()
        score = result.get('score', '')
        
        if result.get('company'):
            conf_str = f" ({confidence})" if confidence else ""
            score_str = f" [Score: {score}]" if score else ""
            print(f"{status.ljust(15)} {result['url']} -> {result['company']}{conf_str}{score_str}")
        else:
            error_msg = result.get('error', '')
            print(f"{status.ljust(15)} {result['url']} -> NOT FOUND ({error_msg})")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())