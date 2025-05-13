# Deployment Guide: CarePay Bot

This guide will walk you through deploying the CarePay Bot application to GitHub and then to a Google Cloud Platform (GCP) VM instance.

## 1. Push to GitHub

### 1.1 Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in to your account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Name your repository (e.g., "carepay-bot")
4. Choose visibility (public or private)
5. Do not initialize with README, .gitignore, or license
6. Click "Create repository"

### 1.2 Initialize Git and Push to GitHub

```bash
# Navigate to your project root
cd /path/to/CarePay_Bot

# Initialize Git repository
git init

# Add all files to staging
git add .

# Commit the files
git commit -m "Initial commit"

# Add the GitHub repository as a remote
git remote add origin https://github.com/NavalEP/isha_bot.git

# Push to GitHub
git push -u origin main
```

## 2. Deploy to GCP VM Instance

### 2.1 Create a VM Instance on GCP

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to Compute Engine > VM Instances
4. Click "Create Instance"
5. Configure your VM:
   - Name: carepay-bot
   - Region: Choose a region close to your users
   - Machine type: e2-medium (2 vCPU, 4 GB memory) or higher
   - Boot disk: Ubuntu 20.04 LTS
   - Allow HTTP and HTTPS traffic
6. Click "Create"

### 2.2 Connect to the VM via SSH

```bash
# Connect to your VM using the SSH button in the GCP console
# Or use gcloud command:
gcloud compute ssh carepay-bot --project your-project-id
```

### 2.3 Set Up the VM Environment

```bash
# Update package lists
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv nginx git

# Install Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install PostgreSQL if you're hosting the database on the same VM
sudo apt install -y postgresql postgresql-contrib

# Install certbot for HTTPS
sudo apt install -y certbot python3-certbot-nginx
```

### 2.4 Clone the Repository

```bash
# Clone your repository
git clone https://github.com/yourusername/carepay-bot.git
cd carepay-bot
```

### 2.5 Set Up the Backend

```bash
# Navigate to the backend directory
cd cpapp_backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp sample.env .env

# Edit the .env file with your production settings
nano .env

# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

# Create a superuser (follow the prompts)
python manage.py createsuperuser

# Test the server
python manage.py runserver 0.0.0.0:8000

# Press Ctrl+C to stop the test server
```

### 2.6 Set Up Gunicorn for the Backend

```bash
# Install Gunicorn
pip install gunicorn

# Create a systemd service file
sudo nano /etc/systemd/system/carepay-backend.service
```

Add the following content to the service file:

```
[Unit]
Description=CarePay Backend Service
After=network.target

[Service]
User=your-username
Group=www-data
WorkingDirectory=/home/your-username/carepay-bot/cpapp_backend
ExecStart=/home/your-username/carepay-bot/cpapp_backend/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:8000 backend.wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Start and enable the service
sudo systemctl start carepay-backend
sudo systemctl enable carepay-backend
sudo systemctl status carepay-backend
```

### 2.7 Set Up the Frontend

```bash
# Navigate to the frontend directory
cd ~/carepay-bot/Frontend

# Create .env file
cp sample.env .env

# Edit the .env file with your production settings
nano .env

# Install dependencies
npm install

# Build the application
npm run build

# Install PM2 to manage the Node.js process
sudo npm install -g pm2

# Start the Next.js application with PM2
pm2 start npm --name "carepay-frontend" -- start

# Make PM2 start on boot
pm2 startup
sudo env PATH=$PATH:/usr/bin pm2 startup systemd -u your-username --hp /home/your-username
pm2 save
```

### 2.8 Configure Nginx as a Reverse Proxy

```bash
# Create an Nginx configuration file
sudo nano /etc/nginx/sites-available/carepay-bot
```

Add the following content:

```
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /admin {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /static {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /media {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

```bash
# Create a symbolic link to enable the site
sudo ln -s /etc/nginx/sites-available/carepay-bot /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 2.9 Set Up HTTPS with Let's Encrypt

```bash
# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Follow the prompts to complete the setup
```

### 2.10 Set Up Automatic Updates and Deployment

Create a deployment script:

```bash
# Create a deployment script
nano ~/deploy.sh
```

Add the following content:

```bash
#!/bin/bash

cd ~/carepay-bot
git pull

# Update backend
cd cpapp_backend
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
sudo systemctl restart carepay-backend

# Update frontend
cd ~/carepay-bot/Frontend
npm install
npm run build
pm2 restart carepay-frontend
```

```bash
# Make the script executable
chmod +x ~/deploy.sh

# Set up a cron job to check for updates (optional)
crontab -e
```

Add the following line to run the deployment script daily at midnight:

```
0 0 * * * ~/deploy.sh >> ~/deploy.log 2>&1
```

## 3. Monitoring and Maintenance

### 3.1 Monitor Logs

```bash
# Backend logs
sudo journalctl -u carepay-backend

# Frontend logs
pm2 logs carepay-frontend

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 3.2 Backup Database

```bash
# For PostgreSQL
pg_dump -U your-db-user -h your-db-host -d your-db-name > backup_$(date +%Y%m%d).sql
```

### 3.3 Update the Application

```bash
# Run the deployment script
~/deploy.sh
```

## 4. Troubleshooting

### 4.1 Check Service Status

```bash
# Check backend service
sudo systemctl status carepay-backend

# Check frontend service
pm2 status

# Check Nginx service
sudo systemctl status nginx
```

### 4.2 Restart Services

```bash
# Restart backend
sudo systemctl restart carepay-backend

# Restart frontend
pm2 restart carepay-frontend

# Restart Nginx
sudo systemctl restart nginx
```

### 4.3 Check Firewall Settings

```bash
# Check if firewall is allowing traffic on ports 80 and 443
sudo ufw status
```

If needed, allow these ports:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
``` 
