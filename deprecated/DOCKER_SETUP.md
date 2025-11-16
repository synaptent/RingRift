# Docker Setup Guide for RingRift

This guide will help you install and configure Docker to run the RingRift application and AI service.

## Installing Docker on macOS

### Option 1: Docker Desktop (Recommended for Beginners)

1. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop
   - Click "Download for Mac"
   - Choose the appropriate version:
     - **Apple Silicon (M1/M2/M3)**: Download "Mac with Apple chip"
     - **Intel**: Download "Mac with Intel chip"

2. **Install Docker Desktop:**
   - Open the downloaded `.dmg` file
   - Drag Docker to Applications folder
   - Launch Docker from Applications
   - Follow the setup wizard (sign in optional)

3. **Verify Installation:**
   ```bash
   docker --version
   docker compose version
   ```

4. **Start Docker:**
   - Docker Desktop should start automatically
   - Look for the Docker whale icon in your menu bar
   - When the icon is steady, Docker is running

### Option 2: OrbStack (Lightweight Alternative)

OrbStack is a fast, lightweight Docker alternative for macOS:

1. **Install via Homebrew:**
   ```bash
   brew install orbstack
   ```

2. **Launch OrbStack:**
   ```bash
   open -a OrbStack
   ```

3. **Verify:**
   ```bash
   docker --version
   ```

### Option 3: Colima (Command-Line Only)

For a minimal, command-line only setup:

1. **Install Colima:**
   ```bash
   brew install colima docker docker-compose
   ```

2. **Start Colima:**
   ```bash
   colima start
   ```

3. **Verify:**
   ```bash
   docker --version
   docker compose version
   ```

## Running RingRift with Docker

### Method 1: Run AI Service Only

```bash
# Build and start just the AI service
docker compose up ai-service

# Or in detached mode (background)
docker compose up -d ai-service
```

The AI service will be available at:
- API: http://localhost:8001
- Documentation: http://localhost:8001/docs
- Health check: http://localhost:8001/health

### Method 2: Run Full Stack

```bash
# Start all services (app, ai-service, postgres, redis, etc.)
docker compose up

# Or in detached mode
docker compose up -d
```

### Method 3: Build and Run Separately

```bash
# Build the AI service image
docker compose build ai-service

# Start it
docker compose up ai-service
```

## Common Docker Commands

### Viewing Logs
```bash
# View logs from all services
docker compose logs

# Follow logs in real-time
docker compose logs -f

# View specific service logs
docker compose logs ai-service
docker compose logs -f ai-service
```

### Stopping Services
```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v

# Stop specific service
docker compose stop ai-service
```

### Rebuilding After Code Changes
```bash
# Rebuild and restart
docker compose up --build ai-service

# Force rebuild without cache
docker compose build --no-cache ai-service
```

### Checking Service Status
```bash
# List running containers
docker compose ps

# View resource usage
docker stats
```

### Accessing Container Shell
```bash
# Get a shell inside the AI service container
docker compose exec ai-service /bin/bash

# Or if container is not running
docker compose run ai-service /bin/bash
```

## Troubleshooting

### Docker Desktop Won't Start
- Check if another virtualization tool is running (VirtualBox, VMware)
- Restart your Mac
- Check System Preferences > Privacy & Security for blocked items
- Uninstall and reinstall Docker Desktop

### "Cannot connect to Docker daemon"
```bash
# Check if Docker is running
docker info

# If using Colima, make sure it's started
colima start

# If using Docker Desktop, check the menu bar icon
```

### Port Already in Use
```bash
# Find what's using port 8001
lsof -i :8001

# Kill the process
kill -9 <PID>

# Or use a different port in docker-compose.yml
```

### Permission Errors
```bash
# Add your user to docker group (may require restart)
sudo dscl . append /Groups/docker GroupMembership $(whoami)
```

### Image Won't Build
```bash
# Clear Docker build cache
docker builder prune

# Remove all unused images
docker image prune -a

# Start fresh
docker compose down -v
docker compose build --no-cache
docker compose up
```

### Service Health Check Failing
```bash
# Check service logs
docker compose logs ai-service

# Verify Python dependencies
docker compose exec ai-service pip list

# Test health endpoint manually
curl http://localhost:8001/health
```

## Performance Tips

### Speed Up Builds
Add to `.dockerignore`:
```
node_modules
venv
__pycache__
*.pyc
.git
.env
```

### Limit Resources
Edit `docker-compose.yml` to adjust memory limits:
```yaml
ai-service:
  deploy:
    resources:
      limits:
        memory: 512M  # Adjust as needed
```

### Use BuildKit
Enable faster builds:
```bash
export DOCKER_BUILDKIT=1
docker compose build
```

## Development Workflow

### Recommended Setup

1. **Use Docker Compose for development:**
   ```bash
   docker compose up -d
   ```

2. **Watch logs in separate terminal:**
   ```bash
   docker compose logs -f ai-service
   ```

3. **Make code changes** - Hot reload is enabled
   
4. **Rebuild only when changing dependencies:**
   ```bash
   docker compose up --build ai-service
   ```

### Alternative: Hybrid Setup

Run some services in Docker, others locally:

```bash
# Start only database and cache in Docker
docker compose up -d postgres redis

# Run AI service locally
cd ai-service
./setup.sh  # One-time setup
./run.sh    # Start service

# Run main app with npm
npm run dev
```

## Next Steps

Once Docker is running:

1. **Test the AI service:**
   ```bash
   docker compose up ai-service
   # Visit http://localhost:8001/docs
   ```

2. **Run full stack:**
   ```bash
   docker compose up
   # Main app: http://localhost:3000
   # AI service: http://localhost:8001
   ```

3. **Check the logs:**
   ```bash
   docker compose logs -f
   ```

## Additional Resources

- [Docker Desktop Documentation](https://docs.docker.com/desktop/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [OrbStack Documentation](https://orbstack.dev/docs)
- [Colima Documentation](https://github.com/abiosoft/colima)

## Need Help?

- Run `docker compose logs ai-service` to see error messages
- Check `ai-service/README.md` for service-specific docs
- Ensure all required ports are available (3000, 3001, 8001, 5432, 6379)
