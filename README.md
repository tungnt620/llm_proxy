# LLM proxy

This application act like a proxy for the LLM API.
It provides some features like:
- Authentication & authorization
- Execute query to the LLM API
- Rate limiting
- Payment:
  - Subscription and Billing
  - Trial period
  - Payment gateway integration
- User management
- Usage history

## Installation
- Create postgres database
```bash
psql
CREATE DATABASE llm_proxy;
CREATE USER llm_proxy_user WITH PASSWORD 'llm_proxy_password';
GRANT ALL PRIVILEGES ON DATABASE llm_proxy TO llm_proxy_user;
```

- Create .env file
```bash
cp .env.example .env
```
Update DB_URL, OPENAI_API_KEY to .env file

- Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Development
```bash

```

## Deployment
- First time


- Update
```bash

```

## CI/CD


## Monitoring


## Troubleshooting