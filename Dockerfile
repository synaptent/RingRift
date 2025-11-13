# Multi-stage build for production optimization
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY src/ ./src/

# Build the application
RUN npm run build:server

# Production stage
FROM node:18-alpine AS runtime

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S ringrift -u 1001

# Set working directory
WORKDIR /app

# Copy built application and dependencies
COPY --from=builder --chown=ringrift:nodejs /app/dist ./dist
COPY --from=builder --chown=ringrift:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=ringrift:nodejs /app/package*.json ./

# Create necessary directories
RUN mkdir -p uploads logs && \
    chown -R ringrift:nodejs uploads logs

# Switch to non-root user
USER ringrift

# Expose ports
EXPOSE 3000 3001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

# Start the application
CMD ["node", "dist/server/index.js"]