"""
Xeno Search Engine - Seed Database
Comprehensive list of crawlable websites organized by topic/category

All sites are:
- Server-rendered (crawlable without JavaScript)
- Authoritative sources
- Regularly updated
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re


@dataclass
class SeedSite:
    """A seed website for crawling"""
    url: str
    name: str
    description: str
    max_pages: int = 100  # Default crawl depth
    priority: int = 1  # Higher = crawl first


@dataclass
class Category:
    """A topic category with associated keywords and seed sites"""
    name: str
    keywords: List[str]  # Keywords that trigger this category
    sites: List[SeedSite]
    subcategories: List[str] = field(default_factory=list)


# =============================================================================
# SEED DATABASE - Organized by Category
# =============================================================================

SEED_DATABASE: Dict[str, Category] = {

    # =========================================================================
    # GENERAL KNOWLEDGE & REFERENCE
    # =========================================================================

    "reference": Category(
        name="General Knowledge & Reference",
        keywords=[
            "what", "who", "when", "where", "why", "how", "year", "date", "time",
            "today", "current", "now", "definition", "meaning", "explain", "wiki",
            "encyclopedia", "fact", "information", "history", "timeline", "calendar",
            "century", "decade", "month", "day", "age", "era", "period", "2024", "2025",
            "biography", "person", "country", "city", "capital", "population",
            "language", "currency", "flag", "geography", "continent", "ocean",
            "planet", "solar system", "universe", "science", "math", "physics",
            "chemistry", "biology", "element", "compound", "formula"
        ],
        sites=[
            SeedSite("https://en.wikipedia.org/wiki/Main_Page", "Wikipedia", "Free encyclopedia", 1000, 5),
            SeedSite("https://en.wikipedia.org/wiki/Portal:Current_events", "Wikipedia Current Events", "Current events portal", 200, 4),
            SeedSite("https://en.wikipedia.org/wiki/2025", "Wikipedia 2025", "Year 2025 article", 50, 4),
            SeedSite("https://www.britannica.com/", "Britannica", "Encyclopedia Britannica", 500, 4),
            SeedSite("https://www.worldometers.info/", "Worldometers", "Real-time world statistics", 100, 3),
            SeedSite("https://www.timeanddate.com/", "Time and Date", "World clock and calendars", 150, 3),
            SeedSite("https://www.wolframalpha.com/", "Wolfram Alpha", "Computational knowledge", 100, 3),
            SeedSite("https://www.dictionary.com/", "Dictionary.com", "Definitions and meanings", 200, 2),
            SeedSite("https://www.merriam-webster.com/", "Merriam-Webster", "Dictionary and thesaurus", 200, 2),
            SeedSite("https://simple.wikipedia.org/wiki/Main_Page", "Simple Wikipedia", "Simple English Wikipedia", 300, 2),
        ],
        subcategories=["facts", "definitions", "dates", "general_knowledge"]
    ),

    # =========================================================================
    # TECHNOLOGY & PROGRAMMING
    # =========================================================================

    "programming": Category(
        name="Programming & Development",
        keywords=[
            "programming", "coding", "developer", "software", "code", "algorithm",
            "javascript", "python", "java", "rust", "golang", "typescript", "c++",
            "react", "vue", "angular", "nodejs", "django", "flask", "rails",
            "api", "database", "sql", "nosql", "mongodb", "postgres", "mysql",
            "git", "github", "devops", "docker", "kubernetes", "aws", "cloud",
            "frontend", "backend", "fullstack", "web development", "mobile app",
            "machine learning", "ai", "artificial intelligence", "data science",
            "tutorial", "documentation", "framework", "library"
        ],
        sites=[
            SeedSite("https://docs.python.org/3/", "Python Docs", "Official Python documentation", 500, 3),
            SeedSite("https://developer.mozilla.org/en-US/docs/Web/JavaScript", "MDN JavaScript", "Mozilla JavaScript docs", 500, 3),
            SeedSite("https://developer.mozilla.org/en-US/docs/Web/CSS", "MDN CSS", "Mozilla CSS docs", 300, 2),
            SeedSite("https://developer.mozilla.org/en-US/docs/Web/HTML", "MDN HTML", "Mozilla HTML docs", 200, 2),
            SeedSite("https://reactjs.org/docs/", "React Docs", "Official React documentation", 200, 2),
            SeedSite("https://vuejs.org/guide/", "Vue.js Guide", "Vue.js documentation", 150, 2),
            SeedSite("https://docs.djangoproject.com/en/stable/", "Django Docs", "Django documentation", 300, 2),
            SeedSite("https://fastapi.tiangolo.com/", "FastAPI Docs", "FastAPI documentation", 150, 2),
            SeedSite("https://go.dev/doc/", "Go Docs", "Official Go documentation", 200, 2),
            SeedSite("https://doc.rust-lang.org/book/", "Rust Book", "The Rust Programming Language", 150, 2),
            SeedSite("https://www.typescriptlang.org/docs/", "TypeScript Docs", "TypeScript documentation", 200, 2),
            SeedSite("https://nodejs.org/en/docs/", "Node.js Docs", "Node.js documentation", 200, 2),
            SeedSite("https://kubernetes.io/docs/", "Kubernetes Docs", "Kubernetes documentation", 300, 2),
            SeedSite("https://docs.docker.com/", "Docker Docs", "Docker documentation", 300, 2),
            SeedSite("https://www.postgresql.org/docs/current/", "PostgreSQL Docs", "PostgreSQL documentation", 300, 2),
            SeedSite("https://redis.io/docs/", "Redis Docs", "Redis documentation", 150, 2),
            SeedSite("https://stackoverflow.com/questions", "Stack Overflow", "Programming Q&A", 500, 1),
            SeedSite("https://news.ycombinator.com/", "Hacker News", "Tech news and discussions", 200, 2),
            SeedSite("https://dev.to/", "DEV Community", "Developer articles", 300, 1),
            SeedSite("https://www.freecodecamp.org/news/", "freeCodeCamp", "Programming tutorials", 300, 2),
            SeedSite("https://realpython.com/", "Real Python", "Python tutorials", 200, 2),
            SeedSite("https://css-tricks.com/", "CSS-Tricks", "CSS tutorials and guides", 200, 1),
        ],
        subcategories=["web_development", "data_science", "devops"]
    ),

    "tech_news": Category(
        name="Technology News",
        keywords=[
            "tech news", "technology", "startup", "silicon valley", "apple", "google",
            "microsoft", "amazon", "meta", "facebook", "tesla", "elon musk",
            "iphone", "android", "smartphone", "gadget", "tech industry",
            "cybersecurity", "hacking", "data breach", "privacy", "encryption",
            "tech company", "innovation", "digital", "internet", "5g", "chip",
            "semiconductor", "nvidia", "amd", "intel"
        ],
        sites=[
            SeedSite("https://techcrunch.com/", "TechCrunch", "Technology news", 200, 3),
            SeedSite("https://www.theverge.com/tech", "The Verge", "Tech news and reviews", 200, 3),
            SeedSite("https://arstechnica.com/", "Ars Technica", "Technology news and analysis", 200, 3),
            SeedSite("https://www.wired.com/", "Wired", "Tech, science, culture", 200, 2),
            SeedSite("https://www.engadget.com/", "Engadget", "Tech news and reviews", 150, 2),
            SeedSite("https://www.cnet.com/tech/", "CNET", "Tech reviews and news", 200, 2),
            SeedSite("https://www.zdnet.com/", "ZDNet", "Technology news", 150, 2),
            SeedSite("https://www.technologyreview.com/", "MIT Tech Review", "MIT Technology Review", 150, 2),
            SeedSite("https://www.tomshardware.com/", "Tom's Hardware", "Hardware news and reviews", 200, 2),
            SeedSite("https://9to5mac.com/", "9to5Mac", "Apple news", 150, 2),
            SeedSite("https://9to5google.com/", "9to5Google", "Google news", 150, 2),
        ]
    ),

    # =========================================================================
    # NEWS & CURRENT EVENTS
    # =========================================================================

    "general_news": Category(
        name="General News",
        keywords=[
            "news", "breaking news", "current events", "headlines", "world news",
            "international", "national", "local news", "today", "latest",
            "update", "report", "announcement", "press", "media"
        ],
        sites=[
            SeedSite("https://www.reuters.com/", "Reuters", "International news agency", 300, 3),
            SeedSite("https://apnews.com/", "Associated Press", "AP News", 300, 3),
            SeedSite("https://www.bbc.com/news", "BBC News", "British Broadcasting Corporation", 300, 3),
            SeedSite("https://www.npr.org/sections/news/", "NPR", "National Public Radio", 200, 2),
            SeedSite("https://www.theguardian.com/international", "The Guardian", "British news", 250, 2),
            SeedSite("https://www.aljazeera.com/", "Al Jazeera", "International news", 200, 2),
            SeedSite("https://www.dw.com/en/", "Deutsche Welle", "German international broadcaster", 150, 2),
            SeedSite("https://www.france24.com/en/", "France 24", "French international news", 150, 2),
        ]
    ),

    "us_news": Category(
        name="US News",
        keywords=[
            "us news", "america", "united states", "washington", "congress",
            "senate", "house", "federal", "state", "governor", "mayor",
            "american", "usa"
        ],
        sites=[
            SeedSite("https://www.nytimes.com/", "New York Times", "US newspaper of record", 300, 3),
            SeedSite("https://www.washingtonpost.com/", "Washington Post", "DC-based newspaper", 300, 3),
            SeedSite("https://www.usatoday.com/", "USA Today", "National newspaper", 200, 2),
            SeedSite("https://www.latimes.com/", "LA Times", "Los Angeles Times", 200, 2),
            SeedSite("https://www.chicagotribune.com/", "Chicago Tribune", "Chicago newspaper", 150, 2),
            SeedSite("https://www.politico.com/", "Politico", "Political news", 200, 2),
            SeedSite("https://thehill.com/", "The Hill", "Political news", 200, 2),
        ]
    ),

    # =========================================================================
    # BUSINESS & FINANCE
    # =========================================================================

    "finance": Category(
        name="Finance & Business",
        keywords=[
            "finance", "business", "stock", "market", "investment", "trading",
            "wall street", "nasdaq", "dow jones", "s&p", "crypto", "bitcoin",
            "ethereum", "cryptocurrency", "forex", "economy", "gdp", "inflation",
            "interest rate", "federal reserve", "fed", "bank", "earnings",
            "ipo", "merger", "acquisition", "startup funding", "venture capital",
            "revenue", "profit", "quarterly", "financial", "money", "wealth"
        ],
        sites=[
            SeedSite("https://www.bloomberg.com/", "Bloomberg", "Financial news", 300, 3),
            SeedSite("https://www.reuters.com/business/", "Reuters Business", "Business news", 250, 3),
            SeedSite("https://www.wsj.com/", "Wall Street Journal", "Financial newspaper", 300, 3),
            SeedSite("https://www.ft.com/", "Financial Times", "UK financial news", 250, 3),
            SeedSite("https://www.cnbc.com/", "CNBC", "Business news network", 250, 2),
            SeedSite("https://www.marketwatch.com/", "MarketWatch", "Stock market news", 200, 2),
            SeedSite("https://seekingalpha.com/", "Seeking Alpha", "Investment analysis", 200, 2),
            SeedSite("https://www.investopedia.com/", "Investopedia", "Financial education", 300, 2),
            SeedSite("https://finance.yahoo.com/", "Yahoo Finance", "Stock quotes and news", 200, 2),
            SeedSite("https://www.fool.com/", "Motley Fool", "Investment advice", 200, 1),
            SeedSite("https://www.coindesk.com/", "CoinDesk", "Crypto news", 200, 2),
            SeedSite("https://cointelegraph.com/", "Cointelegraph", "Crypto news", 150, 2),
        ]
    ),

    # =========================================================================
    # SCIENCE & RESEARCH
    # =========================================================================

    "science": Category(
        name="Science & Research",
        keywords=[
            "science", "research", "study", "scientist", "discovery", "experiment",
            "physics", "chemistry", "biology", "astronomy", "space", "nasa",
            "medicine", "health", "disease", "vaccine", "treatment", "clinical",
            "environment", "climate", "ecology", "evolution", "genetics", "dna",
            "quantum", "particle", "atom", "molecule", "cell", "neuroscience",
            "brain", "psychology", "journal", "peer review", "academic"
        ],
        sites=[
            SeedSite("https://www.nature.com/", "Nature", "Scientific journal", 300, 3),
            SeedSite("https://www.science.org/", "Science Magazine", "AAAS journal", 250, 3),
            SeedSite("https://www.scientificamerican.com/", "Scientific American", "Science magazine", 200, 2),
            SeedSite("https://www.newscientist.com/", "New Scientist", "Science news", 200, 2),
            SeedSite("https://phys.org/", "Phys.org", "Physics news", 250, 2),
            SeedSite("https://www.space.com/", "Space.com", "Space and astronomy", 200, 2),
            SeedSite("https://www.nasa.gov/news/", "NASA News", "NASA updates", 200, 3),
            SeedSite("https://www.sciencedaily.com/", "ScienceDaily", "Science news aggregator", 300, 2),
            SeedSite("https://www.livescience.com/", "Live Science", "Science news", 200, 2),
            SeedSite("https://www.quantamagazine.org/", "Quanta Magazine", "Math and science", 150, 2),
            SeedSite("https://www.popsci.com/", "Popular Science", "Science for general audience", 150, 1),
        ]
    ),

    "health": Category(
        name="Health & Medicine",
        keywords=[
            "health", "medical", "medicine", "doctor", "hospital", "treatment",
            "disease", "illness", "symptom", "diagnosis", "therapy", "drug",
            "pharmaceutical", "fda", "clinical trial", "vaccine", "cancer",
            "diabetes", "heart", "mental health", "depression", "anxiety",
            "covid", "coronavirus", "pandemic", "infection", "virus", "bacteria",
            "nutrition", "diet", "exercise", "fitness", "wellness", "healthcare"
        ],
        sites=[
            SeedSite("https://www.webmd.com/", "WebMD", "Health information", 300, 2),
            SeedSite("https://www.mayoclinic.org/", "Mayo Clinic", "Medical information", 300, 3),
            SeedSite("https://www.healthline.com/", "Healthline", "Health articles", 250, 2),
            SeedSite("https://www.nih.gov/news-events/", "NIH News", "National Institutes of Health", 200, 3),
            SeedSite("https://www.cdc.gov/", "CDC", "Centers for Disease Control", 250, 3),
            SeedSite("https://www.who.int/news", "WHO News", "World Health Organization", 200, 3),
            SeedSite("https://medlineplus.gov/", "MedlinePlus", "NIH health info", 250, 2),
            SeedSite("https://www.medicalnewstoday.com/", "Medical News Today", "Health news", 200, 2),
            SeedSite("https://www.health.harvard.edu/", "Harvard Health", "Harvard medical info", 200, 2),
            SeedSite("https://www.drugs.com/", "Drugs.com", "Drug information", 200, 2),
        ]
    ),

    # =========================================================================
    # GAMING & ENTERTAINMENT
    # =========================================================================

    "gaming": Category(
        name="Gaming",
        keywords=[
            "gaming", "video game", "game", "gamer", "esports", "playstation",
            "xbox", "nintendo", "switch", "pc gaming", "steam", "ps5", "ps4",
            "rpg", "fps", "mmorpg", "battle royale", "fortnite", "minecraft",
            "call of duty", "gta", "zelda", "mario", "pokemon", "game awards",
            "game review", "gameplay", "twitch", "streamer", "gaming news",
            "game release", "console", "controller", "vr gaming", "indie game"
        ],
        sites=[
            SeedSite("https://www.ign.com/", "IGN", "Gaming news and reviews", 300, 3),
            SeedSite("https://www.gamespot.com/", "GameSpot", "Gaming news", 250, 3),
            SeedSite("https://kotaku.com/", "Kotaku", "Gaming culture", 200, 2),
            SeedSite("https://www.polygon.com/", "Polygon", "Gaming news", 200, 2),
            SeedSite("https://www.eurogamer.net/", "Eurogamer", "European gaming news", 200, 2),
            SeedSite("https://www.pcgamer.com/", "PC Gamer", "PC gaming news", 200, 2),
            SeedSite("https://www.gamesradar.com/", "GamesRadar", "Gaming news", 200, 2),
            SeedSite("https://www.rockpapershotgun.com/", "Rock Paper Shotgun", "PC gaming", 150, 2),
            SeedSite("https://www.nintendolife.com/", "Nintendo Life", "Nintendo news", 150, 2),
            SeedSite("https://www.pushsquare.com/", "Push Square", "PlayStation news", 150, 2),
            SeedSite("https://www.purexbox.com/", "Pure Xbox", "Xbox news", 150, 2),
            SeedSite("https://thegameawards.com/", "The Game Awards", "Game Awards official", 50, 3),
            SeedSite("https://www.vg247.com/", "VG247", "Gaming news", 150, 2),
        ]
    ),

    "movies_tv": Category(
        name="Movies & TV",
        keywords=[
            "movie", "film", "cinema", "hollywood", "tv show", "television",
            "streaming", "netflix", "hbo", "disney", "amazon prime", "hulu",
            "actor", "actress", "director", "oscar", "academy award", "emmy",
            "golden globe", "box office", "trailer", "review", "series",
            "documentary", "animation", "superhero", "marvel", "dc", "star wars"
        ],
        sites=[
            SeedSite("https://www.imdb.com/news/", "IMDB News", "Movie database news", 200, 2),
            SeedSite("https://www.rottentomatoes.com/", "Rotten Tomatoes", "Movie reviews", 200, 2),
            SeedSite("https://www.hollywoodreporter.com/", "Hollywood Reporter", "Entertainment industry", 200, 3),
            SeedSite("https://variety.com/", "Variety", "Entertainment news", 200, 3),
            SeedSite("https://deadline.com/", "Deadline", "Entertainment news", 200, 2),
            SeedSite("https://www.indiewire.com/", "IndieWire", "Indie film news", 150, 2),
            SeedSite("https://collider.com/", "Collider", "Movie and TV news", 150, 2),
            SeedSite("https://screenrant.com/", "Screen Rant", "Movie news", 200, 1),
            SeedSite("https://www.slashfilm.com/", "SlashFilm", "Film news", 150, 2),
            SeedSite("https://www.vulture.com/", "Vulture", "Entertainment news", 150, 2),
        ]
    ),

    "music": Category(
        name="Music",
        keywords=[
            "music", "song", "album", "artist", "band", "concert", "tour",
            "grammy", "billboard", "chart", "spotify", "apple music", "streaming",
            "hip hop", "rap", "rock", "pop", "country", "jazz", "classical",
            "singer", "rapper", "musician", "producer", "record label", "vinyl"
        ],
        sites=[
            SeedSite("https://www.billboard.com/", "Billboard", "Music charts and news", 200, 3),
            SeedSite("https://pitchfork.com/", "Pitchfork", "Music reviews", 200, 2),
            SeedSite("https://www.rollingstone.com/music/", "Rolling Stone Music", "Music news", 200, 2),
            SeedSite("https://www.nme.com/", "NME", "Music news UK", 150, 2),
            SeedSite("https://consequenceofsound.net/", "Consequence", "Music news", 150, 2),
            SeedSite("https://www.stereogum.com/", "Stereogum", "Indie music news", 150, 2),
            SeedSite("https://www.complex.com/music/", "Complex Music", "Hip-hop and music", 150, 2),
        ]
    ),

    # =========================================================================
    # SPORTS
    # =========================================================================

    "sports": Category(
        name="Sports",
        keywords=[
            "sports", "football", "soccer", "basketball", "baseball", "hockey",
            "tennis", "golf", "olympics", "nfl", "nba", "mlb", "nhl", "fifa",
            "premier league", "champions league", "world cup", "super bowl",
            "playoffs", "championship", "score", "game", "match", "athlete",
            "player", "team", "coach", "trade", "draft", "injury", "espn"
        ],
        sites=[
            SeedSite("https://www.espn.com/", "ESPN", "Sports news network", 300, 3),
            SeedSite("https://www.cbssports.com/", "CBS Sports", "Sports news", 250, 2),
            SeedSite("https://www.nbcsports.com/", "NBC Sports", "Sports coverage", 200, 2),
            SeedSite("https://bleacherreport.com/", "Bleacher Report", "Sports news", 200, 2),
            SeedSite("https://theathletic.com/", "The Athletic", "Sports journalism", 200, 2),
            SeedSite("https://www.si.com/", "Sports Illustrated", "Sports magazine", 200, 2),
            SeedSite("https://www.bbc.com/sport", "BBC Sport", "UK sports news", 200, 2),
            SeedSite("https://www.skysports.com/", "Sky Sports", "UK sports", 150, 2),
            SeedSite("https://www.goal.com/", "Goal", "Football/soccer news", 150, 2),
        ]
    ),

    # =========================================================================
    # EDUCATION & REFERENCE
    # =========================================================================

    "education": Category(
        name="Education & Reference",
        keywords=[
            "education", "learn", "course", "tutorial", "university", "college",
            "school", "student", "teacher", "professor", "degree", "academic",
            "research", "study", "exam", "test", "certification", "training",
            "online learning", "mooc", "lecture", "curriculum", "scholarship"
        ],
        sites=[
            SeedSite("https://en.wikipedia.org/wiki/Main_Page", "Wikipedia", "Free encyclopedia", 500, 2),
            SeedSite("https://www.khanacademy.org/", "Khan Academy", "Free education", 300, 2),
            SeedSite("https://www.britannica.com/", "Britannica", "Encyclopedia", 300, 2),
            SeedSite("https://www.edx.org/", "edX", "Online courses", 200, 2),
            SeedSite("https://www.coursera.org/", "Coursera", "Online courses", 200, 2),
            SeedSite("https://ocw.mit.edu/", "MIT OpenCourseWare", "Free MIT courses", 300, 2),
            SeedSite("https://www.ted.com/talks", "TED Talks", "Educational talks", 200, 2),
        ]
    ),

    # =========================================================================
    # POLITICS & GOVERNMENT
    # =========================================================================

    "politics": Category(
        name="Politics & Government",
        keywords=[
            "politics", "government", "election", "vote", "president", "congress",
            "senate", "parliament", "democrat", "republican", "liberal", "conservative",
            "policy", "legislation", "law", "bill", "supreme court", "justice",
            "campaign", "candidate", "primary", "poll", "political", "biden",
            "trump", "governor", "mayor", "diplomat", "foreign policy"
        ],
        sites=[
            SeedSite("https://www.politico.com/", "Politico", "Political news", 250, 3),
            SeedSite("https://thehill.com/", "The Hill", "Congressional news", 200, 2),
            SeedSite("https://www.rollcall.com/", "Roll Call", "Congress news", 150, 2),
            SeedSite("https://fivethirtyeight.com/", "FiveThirtyEight", "Political analysis", 150, 2),
            SeedSite("https://www.realclearpolitics.com/", "RealClearPolitics", "Political news", 150, 2),
            SeedSite("https://www.govtrack.us/", "GovTrack", "Congress tracking", 150, 2),
            SeedSite("https://www.congress.gov/", "Congress.gov", "Official Congress site", 200, 3),
            SeedSite("https://www.whitehouse.gov/", "White House", "Official White House", 100, 3),
        ]
    ),

    # =========================================================================
    # LIFESTYLE & CULTURE
    # =========================================================================

    "food": Category(
        name="Food & Cooking",
        keywords=[
            "food", "recipe", "cooking", "chef", "restaurant", "cuisine",
            "ingredient", "baking", "kitchen", "meal", "dinner", "lunch",
            "breakfast", "dessert", "vegetarian", "vegan", "healthy eating",
            "nutrition", "diet", "food review", "michelin"
        ],
        sites=[
            SeedSite("https://www.seriouseats.com/", "Serious Eats", "Food and recipes", 200, 2),
            SeedSite("https://www.bonappetit.com/", "Bon Appetit", "Food magazine", 200, 2),
            SeedSite("https://www.epicurious.com/", "Epicurious", "Recipes and cooking", 200, 2),
            SeedSite("https://www.foodnetwork.com/", "Food Network", "Cooking shows and recipes", 200, 2),
            SeedSite("https://www.allrecipes.com/", "AllRecipes", "Recipe database", 300, 2),
            SeedSite("https://www.food52.com/", "Food52", "Food community", 150, 2),
            SeedSite("https://www.eater.com/", "Eater", "Restaurant news", 200, 2),
        ]
    ),

    "travel": Category(
        name="Travel",
        keywords=[
            "travel", "vacation", "trip", "flight", "hotel", "destination",
            "tourism", "tourist", "airline", "airport", "beach", "mountain",
            "city guide", "travel guide", "backpacking", "cruise", "resort",
            "passport", "visa", "international travel"
        ],
        sites=[
            SeedSite("https://www.lonelyplanet.com/", "Lonely Planet", "Travel guides", 300, 2),
            SeedSite("https://www.tripadvisor.com/", "TripAdvisor", "Travel reviews", 250, 2),
            SeedSite("https://www.cntraveler.com/", "Conde Nast Traveler", "Luxury travel", 150, 2),
            SeedSite("https://www.travelandleisure.com/", "Travel + Leisure", "Travel magazine", 150, 2),
            SeedSite("https://thepointsguy.com/", "The Points Guy", "Travel rewards", 150, 2),
            SeedSite("https://www.afar.com/", "AFAR", "Travel magazine", 150, 2),
        ]
    ),

    # =========================================================================
    # ENVIRONMENT & CLIMATE
    # =========================================================================

    "environment": Category(
        name="Environment & Climate",
        keywords=[
            "environment", "climate", "global warming", "climate change",
            "renewable energy", "solar", "wind", "sustainability", "carbon",
            "emissions", "pollution", "recycling", "conservation", "wildlife",
            "deforestation", "ocean", "biodiversity", "green energy", "ev",
            "electric vehicle", "clean energy"
        ],
        sites=[
            SeedSite("https://www.theguardian.com/environment", "Guardian Environment", "Environmental news", 200, 2),
            SeedSite("https://www.nature.com/nclimate/", "Nature Climate", "Climate research", 150, 2),
            SeedSite("https://www.epa.gov/newsreleases", "EPA News", "Environmental Protection Agency", 150, 3),
            SeedSite("https://grist.org/", "Grist", "Environmental news", 150, 2),
            SeedSite("https://www.carbonbrief.org/", "Carbon Brief", "Climate science", 150, 2),
            SeedSite("https://insideclimatenews.org/", "Inside Climate News", "Climate journalism", 150, 2),
            SeedSite("https://www.climatecentral.org/", "Climate Central", "Climate news", 100, 2),
        ]
    ),

    # =========================================================================
    # AUTOMOTIVE
    # =========================================================================

    "automotive": Category(
        name="Automotive",
        keywords=[
            "car", "vehicle", "automotive", "auto", "truck", "suv", "sedan",
            "electric car", "ev", "tesla", "ford", "toyota", "honda", "bmw",
            "mercedes", "audi", "porsche", "ferrari", "car review", "mpg",
            "horsepower", "engine", "hybrid", "self-driving", "autonomous"
        ],
        sites=[
            SeedSite("https://www.caranddriver.com/", "Car and Driver", "Car reviews", 200, 2),
            SeedSite("https://www.motortrend.com/", "MotorTrend", "Automotive news", 200, 2),
            SeedSite("https://www.autoblog.com/", "Autoblog", "Car news", 200, 2),
            SeedSite("https://www.edmunds.com/", "Edmunds", "Car reviews and prices", 200, 2),
            SeedSite("https://jalopnik.com/", "Jalopnik", "Car culture", 150, 2),
            SeedSite("https://electrek.co/", "Electrek", "EV news", 150, 2),
            SeedSite("https://insideevs.com/", "InsideEVs", "Electric vehicle news", 150, 2),
        ]
    ),

    # =========================================================================
    # LEGAL
    # =========================================================================

    "legal": Category(
        name="Legal & Law",
        keywords=[
            "law", "legal", "court", "judge", "lawyer", "attorney", "lawsuit",
            "case", "verdict", "trial", "supreme court", "ruling", "legislation",
            "regulation", "compliance", "contract", "intellectual property",
            "patent", "copyright", "criminal", "civil", "constitutional"
        ],
        sites=[
            SeedSite("https://www.law.com/", "Law.com", "Legal news", 150, 2),
            SeedSite("https://www.reuters.com/legal/", "Reuters Legal", "Legal news", 200, 2),
            SeedSite("https://www.scotusblog.com/", "SCOTUSblog", "Supreme Court news", 100, 3),
            SeedSite("https://www.lawfaremedia.org/", "Lawfare", "National security law", 100, 2),
            SeedSite("https://www.findlaw.com/", "FindLaw", "Legal information", 200, 2),
            SeedSite("https://www.justia.com/", "Justia", "Legal resources", 200, 2),
        ]
    ),

    # =========================================================================
    # REAL ESTATE
    # =========================================================================

    "real_estate": Category(
        name="Real Estate",
        keywords=[
            "real estate", "housing", "home", "property", "mortgage", "rent",
            "apartment", "condo", "house price", "housing market", "realtor",
            "buy home", "sell home", "foreclosure", "interest rate mortgage"
        ],
        sites=[
            SeedSite("https://www.realtor.com/news/", "Realtor.com News", "Real estate news", 150, 2),
            SeedSite("https://www.zillow.com/research/", "Zillow Research", "Housing data", 100, 2),
            SeedSite("https://www.curbed.com/", "Curbed", "Real estate news", 150, 2),
            SeedSite("https://www.housingwire.com/", "HousingWire", "Mortgage and housing", 150, 2),
        ]
    ),
}


# =============================================================================
# CATEGORY MATCHING FUNCTIONS
# =============================================================================

def find_matching_categories(query: str, max_categories: int = 3) -> List[str]:
    """
    Find categories that match the given query based on keywords.
    Returns list of category names sorted by relevance.
    """
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))

    category_scores: Dict[str, int] = {}

    for cat_name, category in SEED_DATABASE.items():
        score = 0

        # Check keyword matches
        for keyword in category.keywords:
            keyword_lower = keyword.lower()
            # Exact phrase match (higher score)
            if keyword_lower in query_lower:
                score += 3
            # Word overlap
            keyword_words = set(re.findall(r'\b\w+\b', keyword_lower))
            overlap = len(query_words & keyword_words)
            score += overlap

        if score > 0:
            category_scores[cat_name] = score

    # Sort by score descending
    sorted_categories = sorted(
        category_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [cat for cat, score in sorted_categories[:max_categories]]


def get_sites_for_query(query: str, max_sites: int = 10) -> List[SeedSite]:
    """
    Get the most relevant seed sites for a query.
    """
    matching_categories = find_matching_categories(query)

    if not matching_categories:
        # Default to general news if no match
        matching_categories = ["general_news", "tech_news"]

    all_sites: List[SeedSite] = []

    for cat_name in matching_categories:
        category = SEED_DATABASE.get(cat_name)
        if category:
            all_sites.extend(category.sites)

    # Sort by priority and deduplicate
    seen_urls = set()
    unique_sites = []
    for site in sorted(all_sites, key=lambda s: s.priority, reverse=True):
        if site.url not in seen_urls:
            seen_urls.add(site.url)
            unique_sites.append(site)

    return unique_sites[:max_sites]


def get_all_categories() -> List[Dict]:
    """Get all categories with their metadata"""
    return [
        {
            "name": cat_name,
            "display_name": category.name,
            "keywords_sample": category.keywords[:10],
            "sites_count": len(category.sites)
        }
        for cat_name, category in SEED_DATABASE.items()
    ]


def get_category_sites(category_name: str) -> List[Dict]:
    """Get all sites for a specific category"""
    category = SEED_DATABASE.get(category_name)
    if not category:
        return []

    return [
        {
            "url": site.url,
            "name": site.name,
            "description": site.description,
            "max_pages": site.max_pages,
            "priority": site.priority
        }
        for site in category.sites
    ]


# =============================================================================
# STATISTICS
# =============================================================================

def get_database_stats() -> Dict:
    """Get statistics about the seed database"""
    total_sites = sum(len(cat.sites) for cat in SEED_DATABASE.values())
    total_keywords = sum(len(cat.keywords) for cat in SEED_DATABASE.values())

    return {
        "total_categories": len(SEED_DATABASE),
        "total_sites": total_sites,
        "total_keywords": total_keywords,
        "categories": [
            {
                "name": cat_name,
                "display_name": cat.name,
                "sites": len(cat.sites),
                "keywords": len(cat.keywords)
            }
            for cat_name, cat in SEED_DATABASE.items()
        ]
    }


if __name__ == "__main__":
    # Test the database
    print("Seed Database Statistics:")
    stats = get_database_stats()
    print(f"  Categories: {stats['total_categories']}")
    print(f"  Total Sites: {stats['total_sites']}")
    print(f"  Total Keywords: {stats['total_keywords']}")
    print()

    # Test query matching
    test_queries = [
        "latest game awards 2024",
        "python programming tutorial",
        "stock market news today",
        "climate change effects",
        "nfl football scores"
    ]

    for query in test_queries:
        categories = find_matching_categories(query)
        sites = get_sites_for_query(query, max_sites=3)
        print(f"Query: '{query}'")
        print(f"  Categories: {categories}")
        print(f"  Top sites: {[s.name for s in sites]}")
        print()
