# 🌐 AI Website Scraper

An AI-powered website scraper built with **Python**, **Streamlit**, **Playwright**, and **BeautifulSoup**. The application can crawl websites, manually select pages to scrape, or use AI to intelligently extract structured information from public webpages.

## ✨ Features

- 🌍 Scrape public websites using Playwright
- 🔗 Detect and display same-domain links
- ✅ Manually select one or multiple pages to scrape
- 🤖 AI-powered structured data extraction (OpenAI / Gemini)
- 📄 Extract headings, sections, services, contact details, and public form metadata
- 🔍 Detect login pages and stop crawling automatically
- 📊 Display token usage and estimated API cost (OpenAI)
- ⏱️ Measure total scraping time
- 💾 Export results as:
  - JSON
  - CSV
  - Markdown

---

## 📂 Scraping Modes

### 1. Manual Selection
- Detect links on the homepage.
- Select one or more pages manually.
- Scrape only the selected pages.

### 2. Manual Selection + AI
- Select pages manually.
- AI extracts structured JSON from the selected pages.

### 3. AI Guided Crawl
- AI chooses which same-domain pages to visit based on the user's prompt.
- Stops automatically when reaching:
  - Maximum pages
  - Maximum crawl depth
  - Login or restricted pages

### 4. Whole Website Crawl
- Crawls public pages within the same domain.
- AI extracts structured information after crawling.

---

## 🛠 Technologies

- Python
- Streamlit
- Playwright
- BeautifulSoup4
- OpenAI API
- Google Gemini API
- dotenv

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/k33iiw/website-scraper.git
cd website-scraper
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Playwright browser:

```bash
playwright install chromium
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

### OpenAI

```env
OPENAI_API_KEY=your_openai_key
```

### Gemini (Optional)

```env
GEMINI_API_KEY=your_gemini_key
```

or

```env
GOOGLE_API_KEY=your_gemini_key
```

---

## ▶️ Running the Application

```bash
streamlit run app-combined.py
```

---

## 📊 Output

The application can export:

- Structured JSON
- CSV
- Markdown

It also displays:

- Total scraping time
- Number of pages scraped
- Token usage
- Estimated OpenAI API cost

---

## 🔒 Login Detection

For websites requiring authentication, the scraper automatically stops crawling when a login or restricted-access page is detected.

Detection includes:

- Password fields
- Login or Sign In pages
- HTTP 401 / 403 responses
- Authentication-required messages

The scraper **does not** collect or submit credentials.

---

## ⚠️ Limitations

- Crawls public webpages only.
- Does not bypass authentication.
- Does not scrape private user accounts.
- AI output depends on the selected model and prompt.
- Some websites may block automated scraping.

---

## 📄 License

This project is intended for educational and research purposes only. Users are responsible for complying with website Terms of Service and applicable laws before scraping any website.

---

## 👤 Author

**Kei Wong**

GitHub: https://github.com/k33iiw
