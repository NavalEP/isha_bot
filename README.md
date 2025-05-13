# CarePay Bot

A full-stack application with a Next.js frontend and Django backend.

## Project Structure

- `Frontend/` - Next.js frontend application
- `cpapp_backend/` - Django backend application

## Setup Instructions

### Frontend Setup

1. Navigate to the Frontend directory:
```bash
cd Frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Backend Setup

1. Navigate to the backend directory:
```bash
cd cpapp_backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
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

The backend API will be available at `http://localhost:8000`

## Deployment

See the deployment guide in the documentation for instructions on deploying to GCP. 