# CarePay Bot Backend

This is the backend service for the CarePay Bot application built with Django.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Start the development server:
```bash
python manage.py runserver
```

The server will start at `http://localhost:8000`

## Project Structure

- `backend/` - Main Django project directory
- `api/` - Django app containing API endpoints
- `manage.py` - Django's command-line utility for administrative tasks 