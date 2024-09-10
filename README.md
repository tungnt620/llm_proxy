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
- Create postgres database, at least version 13
```bash
docker run -d \
	--name llm_proxy_db \
    -p 5432:5432 \
	-e POSTGRES_PASSWORD=mysecretpassword \
	-e PGDATA=/var/lib/postgresql/data/pgdata \
	-v ~/tung/Docker/llm_proxy_db:/var/lib/postgresql/data \
	postgres:16.4
```

Login into postgres shell
```bash
psql -U postgres
CREATE DATABASE llm_proxy;
CREATE USER llm_proxy_user WITH PASSWORD 'llm_proxy_password';
GRANT ALL PRIVILEGES ON DATABASE llm_proxy TO llm_proxy_user;
\c llm_proxy; # select database
GRANT ALL PRIVILEGES ON SCHEMA public TO llm_proxy_user;
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
# Migrate database
python3 manage.py migrate

# Run server
python3 manage.py runserver

# Create superuser
python3 manage.py createsuperuser
```

## Deployment
- First time


- Update
```bash

```

## Security checklist

## Performance checklist

## CI/CD


## Monitoring


## Troubleshooting