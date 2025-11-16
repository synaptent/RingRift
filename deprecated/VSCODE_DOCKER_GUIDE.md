# Using VS Code Docker Extension for RingRift

Great news! The Docker extension in VS Code provides a user-friendly graphical interface for managing Docker containers, making it much easier to set up and run the RingRift environment.

## What the Docker Extension Provides

The VS Code Docker extension gives you:
- ✅ **Visual container management** - No command-line needed
- ✅ **One-click builds and runs** - Right-click to build/start/stop
- ✅ **Log viewing** - See container output in VS Code
- ✅ **Image management** - View and manage Docker images
- ✅ **Compose support** - Work with docker-compose.yml files
- ✅ **Integrated debugging** - Debug apps running in containers

## Prerequisites

You still need Docker installed on your system. The Docker extension uses Docker but doesn't replace it.

### Quick Docker Installation

Choose **ONE** of these options:

1. **OrbStack** (Recommended - Lightweight & Fast):
   ```bash
   brew install orbstack
   open -a OrbStack
   ```

2. **Colima** (Command-line friendly):
   ```bash
   brew install colima docker docker-compose
   colima start
   ```

3. **Docker Desktop** (Full-featured GUI):
   - Download from https://www.docker.com/products/docker-desktop
   - Install and launch

Once installed, verify Docker is running:
```bash
docker --version
```

## Using Docker Extension with RingRift

### Option 1: Using Docker Compose (Recommended)

1. **Open the Docker Extension**
   - Click the Docker icon in the VS Code sidebar (whale icon)
   - Or press `Cmd+Shift+P` and search for "Docker: Focus on Docker View"

2. **Locate docker-compose.yml**
   - In the Docker extension sidebar, look for "COMPOSE" section
   - You should see `ringrift` or your docker-compose.yml file listed

3. **Start the AI Service**
   - Right-click on docker-compose.yml or the `ai-service` entry
   - Select "Compose Up"
   - Choose which services to start:
     - Just `ai-service` for AI only
     - All services for the full stack

4. **View Logs**
   - Expand the running containers in the Docker extension
   - Right-click on `ai-service`
   - Select "View Logs"

5. **Stop Services**
   - Right-click on the running service
   - Select "Compose Down"

### Option 2: Build Individual Images

1. **Build the AI Service Image**
   - In VS Code Explorer, navigate to `ai-service/`
   - Right-click on `Dockerfile`
   - Select "Build Image..."
   - Tag it as `ringrift-ai-service:latest`

2. **Run the Container**
   - In Docker extension sidebar, find the image under "IMAGES"
   - Right-click on `ringrift-ai-service`
   - Select "Run"
   - Configure ports: Map container port `8001` to host port `8001`

### Option 3: Using the Docker Extension UI

1. **Containers View**
   - See all running & stopped containers
   - Start/Stop/Restart with right-click
   - Attach shell to inspect containers
   - View container logs in real-time

2. **Images View**
   - See all Docker images on your system
   - Build images from Dockerfiles
   - Push/Pull images
   - Remove unused images

3. **Registries View**
   - Connect to Docker Hub
   - Manage image repositories

## Quick Start with Docker Extension

### Method 1: One-Click Compose Up

1. Open VS Code in the RingRift directory
2. Open Docker extension (sidebar)
3. Right-click `docker-compose.yml` under COMPOSE
4. Click "Compose Up"
5. Select `ai-service`
6. Wait for build & start (first time takes longer)
7. Check logs by right-clicking the running container → "View Logs"

### Method 2: Build & Run Separately

1. **Build Image:**
   - Right-click `ai-service/Dockerfile`
   - "Build Image..."
   - Enter tag: `ringrift-ai-service`

2. **Run Container:**
   - In Docker extension → IMAGES
   - Right-click `ringrift-ai-service`
   - "Run Interactive"
   - Add port mapping: `8001:8001`

## Verifying the Setup

Once the container is running:

1. **Check Container Status**
   - In Docker extension → CONTAINERS
   - Should show `ai-service` with green arrow (running)

2. **View Logs**
   - Right-click container → "View Logs"
   - Should see: "Uvicorn running on http://0.0.0.0:8001"

3. **Test the Service**
   - Open browser: http://localhost:8001/health
   - Should return: `{"status":"healthy"}`
   - API docs: http://localhost:8001/docs

## Commands Available in Docker Extension

Right-click on containers for these options:

- **Start** - Start a stopped container
- **Stop** - Stop a running container
- **Restart** - Restart container
- **Remove** - Delete container
- **View Logs** - See container output
- **Attach Shell** - Open terminal inside container
- **Attach Visual Studio Code** - Open container in VS Code
- **Inspect** - View container details

## Troubleshooting

### "Docker not running"
- Make sure OrbStack, Colima, or Docker Desktop is running
- Check: `docker ps` should work in terminal

### "Cannot find docker-compose.yml"
- Make sure you opened VS Code in the `/Users/armand/code/RingRift` directory
- Reload VS Code: `Cmd+Shift+P` → "Developer: Reload Window"

### "Build fails"
- Check Dockerfile syntax
- View build output in Docker extension logs
- Try building from terminal: `docker build -t ringrift-ai-service ./ai-service`

### "Port 8001 already in use"
- Stop any local Python service: Check if `./run.sh` is running
- Or change port in docker-compose.yml

## Advantages of Using Docker Extension

### Compared to Command Line:
✅ **Visual feedback** - See what's running at a glance  
✅ **Easier management** - Right-click instead of remembering commands  
✅ **Integrated logs** - View logs in VS Code instead of separate terminal  
✅ **Quick access** - Start/stop services with 2 clicks  
✅ **No memorization** - Don't need to remember Docker commands

### Compared to Local Python:
✅ **Isolated environment** - Doesn't affect your system Python  
✅ **Consistent setup** - Same environment for all developers  
✅ **Easy cleanup** - Remove containers without trace  
✅ **Production-ready** - Same setup as deployment  
✅ **All dependencies included** - No pip install issues

## Recommended Workflow

For **development**:

1. **Start AI Service via Docker Extension**
   ```
   Right-click docker-compose.yml → Compose Up → ai-service
   ```

2. **View Logs in VS Code**
   ```
   Right-click ai-service container → View Logs
   ```

3. **Make code changes**
   - Edit Python files in `ai-service/app/`
   - Hot reload is enabled, changes apply automatically

4. **Test via API docs**
   - http://localhost:8001/docs

5. **When done, stop service**
   ```
   Right-click container → Stop
   ```

For **production testing**:

1. Build all services:
   ```
   Right-click docker-compose.yml → Compose Up (select all)
   ```

2. Test full stack integration

3. Check all container logs

## Docker Extension Settings

Recommended settings for VS Code:

1. **Open Settings** (`Cmd+,`)
2. Search for "docker"
3. Recommended configuration:
   ```json
   {
     "docker.commands.build": "${containerCommand} build --pull --rm -f ${dockerfile} -t ${tag} ${context}",
     "docker.commands.run": "${containerCommand} run --rm -d -p 8001:8001 ${tag}",
     "docker.dockerComposeBuild": true,
     "docker.dockerComposeDetached": true,
     "docker.showStartPage": false,
     "docker.containers.groupBy": "Compose Project Name"
   }
   ```

## Next Steps

1. ✅ **Install Docker** (OrbStack, Colima, or Docker Desktop)
2. ✅ **Verify Docker is running** (`docker ps` works)
3. ✅ **Open Docker Extension** in VS Code sidebar
4. ✅ **Right-click docker-compose.yml** → Compose Up
5. ✅ **Select ai-service**
6. ✅ **Wait for build** (first time takes a few minutes)
7. ✅ **Check logs** (should see "Uvicorn running")
8. ✅ **Test API** at http://localhost:8001/docs

## Summary

**Yes, the VS Code Docker Extension makes it MUCH easier to set up the environment!**

Instead of typing Docker commands, you can:
- Build and run containers with right-clicks
- View logs directly in VS Code
- Start/stop services visually
- See what's running at a glance

It's the recommended way to work with Docker in VS Code. You just need to install Docker (OrbStack, Colima, or Docker Desktop) first, then the extension handles the rest through its GUI.
