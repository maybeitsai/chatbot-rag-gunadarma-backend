# modules/crawler.py
"""
Gunadarma Deep Web Crawler
Crawler komprehensif untuk mengekstrak seluruh konten dari situs Gunadarma
"""

import asyncio
import json
import csv
import os
import re
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse, urlunparse, unquote
from typing import Set, List, Dict, Optional
import logging

# Import libraries yang diperlukan
try:
    import requests
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright
    import PyPDF2
    import pdfplumber
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as e:
    print(f"Error: Missing required library. Please install: {e}")
    print("Run: pip install requests beautifulsoup4 playwright PyPDF2 pdfplumber")
    print("Then run: playwright install")
    exit(1)


class DeepCrawler:
    def __init__(self, target_urls: List[str], max_depth: int = 3):
        self.target_urls = target_urls
        self.max_depth = max_depth
        self.visited_urls: Set[str] = set()
        self.crawled_data: List[Dict] = []
        self.pdf_urls: Set[str] = set()
        self.allowed_domains = self._get_allowed_domains()

        # Define file extensions to ignore
        self.ignored_extensions = {
            # Image files
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".svg",
            ".webp",
            ".ico",
            ".tiff",
            ".tif",
            # Video files
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".mkv",
            ".m4v",
            # Audio files
            ".mp3",
            ".wav",
            ".ogg",
            ".m4a",
            ".aac",
            ".flac",
            ".wma",
            # Archive files
            ".zip",
            ".rar",
            ".7z",
            ".tar",
            ".gz",
            ".bz2",
            # Other binary files
            ".exe",
            ".dmg",
            ".pkg",
            ".deb",
            ".rpm",
            # Office files
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("crawler.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        # Setup requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Setup headers untuk bypass Cloudflare
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            }
        )

        # Create data directory
        os.makedirs("data", exist_ok=True)

    def _get_allowed_domains(self) -> Set[str]:
        """Mendapatkan domain yang diizinkan untuk crawling"""
        domains = set()
        for url in self.target_urls:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            domains.add(domain)

            # Tambahkan subdomain gunadarma.ac.id
            if "gunadarma.ac.id" in domain:
                domains.add("gunadarma.ac.id")
                # Izinkan semua subdomain gunadarma
                base_domain = "gunadarma.ac.id"
                domains.add(f"*.{base_domain}")

        return domains

    def _is_allowed_url(self, url: str) -> bool:
        """Mengecek apakah URL diizinkan untuk di-crawl"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Cek domain utama
            for allowed_domain in self.allowed_domains:
                if allowed_domain.startswith("*."):
                    base_domain = allowed_domain[2:]
                    if domain.endswith(base_domain):
                        return True
                elif domain == allowed_domain or domain.endswith("." + allowed_domain):
                    return True

            return False
        except:
            return False

    def _is_ignored_file(self, url: str) -> bool:
        """Mengecek apakah URL mengarah ke file yang harus diabaikan"""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()

            # Check file extension
            for ext in self.ignored_extensions:
                if path.endswith(ext):
                    return True

            # Additional checks for query parameters that might indicate media files
            query = parsed.query.lower()
            if any(param in query for param in ["image", "img", "photo", "pic"]):
                return True

            return False
        except:
            return False

    def _clean_url(self, url: str) -> str:
        """Membersihkan URL dari fragment dan parameter yang tidak perlu"""
        parsed = urlparse(url)
        # Hapus fragment (#)
        cleaned = urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, "")
        )
        return cleaned

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Mengekstrak semua link dari halaman"""
        links = set()

        # Ekstrak dari tag <a>
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            if href and not href.startswith("#"):
                absolute_url = urljoin(base_url, href)
                absolute_url = self._clean_url(absolute_url)

                # Skip if it's an ignored file type
                if self._is_ignored_file(absolute_url):
                    self.logger.debug(f"Skipping ignored file: {absolute_url}")
                    continue

                # Handle PDF files separately
                if absolute_url.lower().endswith(".pdf"):
                    self.pdf_urls.add(absolute_url)
                    continue

                # Only add HTML/web pages
                if self._is_allowed_url(absolute_url):
                    links.add(absolute_url)

        self.logger.info(
            f"Found {len(links)} valid links and {len(self.pdf_urls)} PDF files"
        )
        return links

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Mengekstrak konten teks dari HTML"""
        # Hapus script dan style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Ekstrak teks
        text = soup.get_text()

        # Bersihkan teks
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    async def _crawl_with_playwright(self, url: str) -> Optional[Dict]:
        """Crawl halaman menggunakan Playwright untuk konten dinamis"""
        try:
            async with async_playwright() as p:
                # Gunakan Chrome/Chromium dengan stealth mode
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--disable-extensions",
                        "--disable-gpu",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor",
                    ],
                )

                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                )

                page = await context.new_page()

                # Set additional headers untuk bypass Cloudflare
                await page.set_extra_http_headers(
                    {
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
                        "Accept-Encoding": "gzip, deflate, br",
                        "DNT": "1",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1",
                    }
                )

                # Hapus webdriver properties untuk stealth
                await page.add_init_script(
                    """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                    });
                    
                    // Remove automation indicators
                    delete window.navigator.__proto__.webdriver;
                    
                    // Override permissions
                    Object.defineProperty(navigator, 'permissions', {
                        get: () => ({
                            query: () => Promise.resolve({ state: 'granted' })
                        })
                    });
                    
                    // Override plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    
                    // Override languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en', 'id']
                    });
                """
                )

                self.logger.info(
                    f"Attempting to load {url} with enhanced stealth mode..."
                )

                # Navigate dengan timeout yang lebih panjang untuk Cloudflare
                try:
                    response = await page.goto(
                        url, wait_until="domcontentloaded", timeout=60000
                    )

                    if response:
                        self.logger.info(f"Response status: {response.status}")

                    # Tunggu lebih lama untuk Cloudflare challenge
                    await page.wait_for_timeout(5000)

                    # Cek apakah ada Cloudflare challenge
                    cloudflare_selectors = [
                        "[data-ray]",
                        ".cf-browser-verification",
                        "#cf-challenge-running",
                        ".challenge-running",
                        'div[class*="cloudflare"]',
                    ]

                    for selector in cloudflare_selectors:
                        try:
                            element = await page.query_selector(selector)
                            if element:
                                self.logger.info(
                                    "Cloudflare challenge detected, waiting..."
                                )
                                await page.wait_for_timeout(10000)  # Tunggu 10 detik
                                break
                        except:
                            continue

                    # Tunggu sampai halaman fully loaded
                    await page.wait_for_load_state("networkidle", timeout=30000)

                    # Dapatkan konten
                    content = await page.content()
                    title = await page.title()

                    # Cek apakah konten valid (bukan halaman error Cloudflare)
                    if "cloudflare" in content.lower() and len(content) < 5000:
                        self.logger.warning(
                            f"Possible Cloudflare block detected for {url}"
                        )
                        # Coba sekali lagi dengan delay tambahan
                        await page.wait_for_timeout(5000)
                        content = await page.content()
                        title = await page.title()

                    await browser.close()

                    return {
                        "url": url,
                        "title": title,
                        "content": content,
                        "source_type": "html",
                        "timestamp": datetime.now().isoformat(),
                    }

                except Exception as nav_error:
                    self.logger.error(f"Navigation error for {url}: {str(nav_error)}")
                    await browser.close()
                    return None

        except Exception as e:
            self.logger.error(f"Playwright crawl error for {url}: {str(e)}")
            return None

    def _crawl_with_requests(self, url: str) -> Optional[Dict]:
        """Crawl halaman menggunakan requests dengan enhanced Cloudflare bypass"""
        try:
            # Tambahan headers khusus untuk Cloudflare
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "Referer": "https://www.google.com/",
            }

            # Update session headers
            self.session.headers.update(headers)

            response = self.session.get(url, timeout=60)

            # Cek apakah ada Cloudflare challenge
            if response.status_code == 503 or "cloudflare" in response.text.lower():
                self.logger.warning(
                    f"Cloudflare protection detected for {url}, switching to Playwright"
                )
                return None

            response.raise_for_status()

            return {
                "url": url,
                "title": "",
                "content": response.text,
                "source_type": "html",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Requests crawl error for {url}: {str(e)}")
            return None

    def _extract_pdf_content(self, pdf_url: str) -> Optional[str]:
        """Mengekstrak teks dari file PDF dengan improved handling"""
        try:
            self.logger.info(f"Downloading PDF: {pdf_url}")

            # Setup headers khusus untuk PDF
            pdf_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/pdf,application/octet-stream,*/*",
                "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "Referer": "https://baak.gunadarma.ac.id/",
            }

            # Create new session for PDF to avoid conflicts
            pdf_session = requests.Session()
            pdf_session.headers.update(pdf_headers)

            # Setup retry for PDF session
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            pdf_session.mount("http://", adapter)
            pdf_session.mount("https://", adapter)

            response = pdf_session.get(pdf_url, timeout=60, stream=True)

            self.logger.info(f"PDF response status: {response.status_code}")

            if response.status_code != 200:
                self.logger.error(
                    f"Failed to download PDF. Status: {response.status_code}"
                )
                return None

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            self.logger.info(f"PDF content type: {content_type}")

            if "pdf" not in content_type and "octet-stream" not in content_type:
                if "html" in content_type:
                    self.logger.error(
                        "Received HTML instead of PDF - likely blocked by Cloudflare"
                    )
                    return None
                else:
                    self.logger.warning(
                        f"Unexpected content type for PDF: {content_type}"
                    )

            # Simpan PDF sementara dengan nama yang aman
            parsed_url = urlparse(pdf_url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.endswith(".pdf"):
                filename = f"document_{int(time.time())}.pdf"

            # Clean filename
            filename = "".join(
                c for c in filename if c.isalnum() or c in (" ", "-", "_", ".")
            ).rstrip()
            pdf_path = f"temp_{int(time.time())}_{filename}"

            # Download dengan streaming
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Check file size
            file_size = os.path.getsize(pdf_path)
            self.logger.info(f"Downloaded PDF size: {file_size} bytes")

            if file_size < 1000:  # Kurang dari 1KB mencurigakan
                self.logger.warning("PDF file is very small, might be an error page")
                os.remove(pdf_path)
                return None

            text_content = ""

            # Coba dengan pdfplumber terlebih dahulu
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    self.logger.info(f"PDF has {len(pdf.pages)} pages")
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"--- Page {i+1} ---\n{page_text}\n\n"

                if text_content.strip():
                    self.logger.info(
                        f"Successfully extracted {len(text_content)} characters using pdfplumber"
                    )

            except Exception as e:
                self.logger.warning(f"pdfplumber failed: {str(e)}, trying PyPDF2...")

                # Fallback ke PyPDF2
                try:
                    with open(pdf_path, "rb") as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        self.logger.info(
                            f"PDF has {len(pdf_reader.pages)} pages (PyPDF2)"
                        )

                        for i, page in enumerate(pdf_reader.pages):
                            page_text = page.extract_text()
                            if page_text:
                                text_content += f"--- Page {i+1} ---\n{page_text}\n\n"

                    if text_content.strip():
                        self.logger.info(
                            f"Successfully extracted {len(text_content)} characters using PyPDF2"
                        )

                except Exception as e2:
                    self.logger.error(
                        f"Both PDF extraction methods failed: pdfplumber({e}), PyPDF2({e2})"
                    )

            # Hapus file sementara
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

            return text_content.strip() if text_content.strip() else None

        except Exception as e:
            self.logger.error(f"PDF extraction error for {pdf_url}: {str(e)}")
            return None

    async def _crawl_page(self, url: str, depth: int = 0) -> Set[str]:
        """Crawl satu halaman dan return link yang ditemukan"""
        if url in self.visited_urls or depth > self.max_depth:
            return set()

        # Skip if it's an ignored file type
        if self._is_ignored_file(url):
            self.logger.debug(f"Skipping ignored file: {url}")
            return set()

        self.visited_urls.add(url)
        self.logger.info(f"Crawling (depth {depth}): {url}")

        # Check jika ini adalah PDF file
        if url.lower().endswith(".pdf"):
            self.logger.info(f"PDF detected: {url}")
            self.pdf_urls.add(url)
            return set()  # PDF tidak memiliki link, langsung return

        page_data = None

        # Untuk situs BAAK yang menggunakan Cloudflare, langsung gunakan Playwright
        if "baak.gunadarma.ac.id" in url:
            self.logger.info(
                "BAAK site detected - using Playwright with Cloudflare bypass"
            )
            page_data = await self._crawl_with_playwright(url)
        else:
            # Coba crawl dengan requests terlebih dahulu untuk situs lain
            page_data = self._crawl_with_requests(url)

            # Jika gagal, coba dengan Playwright
            if not page_data:
                self.logger.info("Requests failed, trying Playwright...")
                page_data = await self._crawl_with_playwright(url)

        if not page_data:
            self.logger.error(f"Failed to crawl {url} with both methods")
            return set()

        # Parse HTML
        soup = BeautifulSoup(page_data["content"], "html.parser")

        # Dapatkan title jika belum ada
        if not page_data["title"] and soup.title:
            page_data["title"] = soup.title.string.strip() if soup.title.string else ""

        # Ekstrak konten teks
        text_content = self._extract_text_content(soup)

        # Validasi konten (pastikan bukan halaman error Cloudflare)
        if len(text_content.strip()) < 100 and "cloudflare" in text_content.lower():
            self.logger.warning(
                f"Suspected Cloudflare block for {url}, content too short"
            )
            return set()

        # Simpan data
        crawl_result = {
            "url": url,
            "title": page_data["title"],
            "text_content": text_content,
            "source_type": "html",
            "timestamp": page_data["timestamp"],
        }

        self.crawled_data.append(crawl_result)
        self.logger.info(
            f"Successfully crawled {url} - Content length: {len(text_content)} chars"
        )

        # Ekstrak link untuk crawling selanjutnya
        links = self._extract_links(soup, url)

        return links

    async def crawl_all_pages(self):
        """Melakukan crawling pada semua halaman secara rekursif"""
        self.logger.info("Starting comprehensive crawling...")

        # Queue untuk URL yang akan di-crawl
        url_queue = [(url, 0) for url in self.target_urls]

        while url_queue:
            current_url, depth = url_queue.pop(0)

            if current_url not in self.visited_urls and depth <= self.max_depth:
                try:
                    # Crawl halaman dan dapatkan link baru
                    new_links = await self._crawl_page(current_url, depth)

                    # Tambahkan link baru ke queue
                    for link in new_links:
                        if link not in self.visited_urls:
                            url_queue.append((link, depth + 1))

                    # Delay untuk menghindari rate limiting (lebih lama untuk BAAK)
                    if "baak.gunadarma.ac.id" in current_url:
                        await asyncio.sleep(3)
                    else:
                        await asyncio.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error crawling {current_url}: {str(e)}")

        self.logger.info(f"Crawled {len(self.crawled_data)} pages")

    async def process_pdfs(self):
        """Memproses semua file PDF yang ditemukan"""
        self.logger.info(f"Processing {len(self.pdf_urls)} PDF files...")

        for pdf_url in self.pdf_urls:
            try:
                self.logger.info(f"Extracting PDF: {pdf_url}")
                text_content = self._extract_pdf_content(pdf_url)

                if text_content:
                    pdf_result = {
                        "url": pdf_url,
                        "title": os.path.basename(urlparse(pdf_url).path),
                        "text_content": text_content,
                        "source_type": "pdf",
                        "timestamp": datetime.now().isoformat(),
                    }

                    self.crawled_data.append(pdf_result)
                    self.logger.info(f"Successfully processed PDF: {pdf_url}")

                # Delay untuk PDF processing
                await asyncio.sleep(2)

            except Exception as e:
                self.logger.error(f"Error processing PDF {pdf_url}: {str(e)}")

    def save_results(self):
        """Menyimpan hasil crawling ke file JSON dan CSV"""
        self.logger.info("Saving results...")

        # Simpan ke JSON
        json_path = "data/output.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.crawled_data, f, indent=2, ensure_ascii=False)

        # Simpan ke CSV
        csv_path = "data/output.csv"
        if self.crawled_data:
            fieldnames = ["url", "title", "text_content", "source_type", "timestamp"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.crawled_data)

        self.logger.info(f"Results saved:")
        self.logger.info(f"- JSON: {json_path} ({len(self.crawled_data)} entries)")
        self.logger.info(f"- CSV: {csv_path}")

    def print_summary(self):
        """Mencetak ringkasan hasil crawling"""
        html_count = sum(
            1 for item in self.crawled_data if item["source_type"] == "html"
        )
        pdf_count = sum(1 for item in self.crawled_data if item["source_type"] == "pdf")

        print("\n" + "=" * 60)
        print("CRAWLING SUMMARY")
        print("=" * 60)
        print(f"Total pages crawled: {len(self.crawled_data)}")
        print(f"HTML pages: {html_count}")
        print(f"PDF files: {pdf_count}")
        print(f"Unique URLs visited: {len(self.visited_urls)}")
        print(f"Maximum depth reached: {self.max_depth}")

        # Show ignored file statistics
        ignored_count = (
            len(self.visited_urls) - len(self.crawled_data) - len(self.pdf_urls)
        )
        if ignored_count > 0:
            print(f"Ignored files (images, media, etc.): {ignored_count}")

        print("\nDomains crawled:")

        domains = set()
        for item in self.crawled_data:
            domain = urlparse(item["url"]).netloc
            domains.add(domain)

        for domain in sorted(domains):
            domain_count = sum(
                1
                for item in self.crawled_data
                if urlparse(item["url"]).netloc == domain
            )
            print(f"  - {domain}: {domain_count} pages")

        print("=" * 60)


async def crawl_pipeline():
    """Fungsi utama untuk menjalankan crawler"""
    target_urls = ["https://baak.gunadarma.ac.id/", "https://www.gunadarma.ac.id/"]

    print("Starting Gunadarma Deep Web Crawler...")
    print(f"Target URLs: {target_urls}")
    print(f"Output directory: ./data/")

    # Inisialisasi crawler
    crawler = DeepCrawler(target_urls, max_depth=2)

    try:
        # Lakukan crawling
        start_time = time.time()

        await crawler.crawl_all_pages()
        await crawler.process_pdfs()

        # Simpan hasil
        crawler.save_results()

        # Tampilkan ringkasan
        crawler.print_summary()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nCrawling completed in {duration:.2f} seconds")
        print("Results saved to:")
        print("  - data/output.json")
        print("  - data/output.csv")

    except KeyboardInterrupt:
        print("\nCrawling interrupted by user")
        if crawler.crawled_data:
            crawler.save_results()
            print("Partial results saved")

    except Exception as e:
        print(f"Critical error: {str(e)}")
        if crawler.crawled_data:
            crawler.save_results()
            print("Partial results saved")
