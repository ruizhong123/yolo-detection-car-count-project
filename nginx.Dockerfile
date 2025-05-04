# Use the official NGINX image as the base
FROM nginx:latest

# Copy custom configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy SSL certificates
COPY project/certs/nginx.crt /etc/nginx/ssl/nginx.crt
COPY project/certs/nginx.key /etc/nginx/ssl/nginx.key
