# AI Question Answering API with Multi-Agent Validation

A REST API that processes multiple-choice questions using a dual-agent AI validation system powered by Google's Gemini 2.5 Pro.

## Project Structure

```
.
├── api/                    # REST API layer
├── agents/                 # AI agent components (Answerer & Validator)
├── workers/                # Concurrent processing worker pool
├── models/                 # Data models and schemas
├── tests/                  # Test suite
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── .env.example           # Environment configuration template
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

3. Run the API server:
```bash
python main.py
```

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `API_PORT`: Port for the API server (default: 8000)
- `MAX_CONCURRENT_WORKERS`: Maximum concurrent question processing workers (default: 5)
- `MAX_VALIDATION_ITERATIONS`: Maximum validation loop iterations (default: 5)
- `LOG_LEVEL`: Logging level (default: INFO)

## API Usage

### POST /api/answer-questions

Submit a list of multiple-choice questions for processing.

**Request Body:**
```json
[
  {
    "content": "Question text with context",
    "title": "Brief question summary",
    "type": "option",
    "answer": ["Option A", "Option B", "Option C", "Option D"],
    "questionNumber": "1"
  }
]
```

**Response:**
```json
[
  {
    "questionNumber": "1",
    "selectedAnswer": "Option B",
    "validationIterations": 2,
    "processingTimeMs": 1234
  }
]
```

## Development

Run tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=. --cov-report=html
```

## Docker Deployment

(Docker configuration will be added in later tasks)
