/**
 * PM2 Ecosystem Configuration for RingRift Production
 *
 * Features:
 * - Structured process management with memory limits
 * - Log rotation (100MB per file, 10 files retained)
 * - Pre-startup port cleanup for AI service
 * - Health check integration
 */
module.exports = {
  apps: [
    {
      name: 'ringrift-server',
      script: 'dist/index.js',
      cwd: '/home/ubuntu/RingRift/src/server',
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      max_memory_restart: '2G',
      env: {
        NODE_ENV: 'production',
        ENABLE_HEALTH_POLLING: 'true',
      },
      // Health check configuration
      listen_timeout: 10000,
      kill_timeout: 5000,
      wait_ready: true,
      // Log rotation
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/home/ubuntu/logs/ringrift-server-error.log',
      out_file: '/home/ubuntu/logs/ringrift-server-out.log',
      merge_logs: true,
      max_size: '100M',
      retain: 10,
    },
    {
      name: 'ringrift-ai',
      script: '/home/ubuntu/RingRift/src/server/scripts/pm2-ai-launcher.sh',
      cwd: '/home/ubuntu/RingRift/ai-service',
      interpreter: 'bash',
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      max_memory_restart: '8G',
      // Health check via HTTP
      listen_timeout: 30000,
      kill_timeout: 10000,
      // Log rotation
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/home/ubuntu/logs/ringrift-ai-error.log',
      out_file: '/home/ubuntu/logs/ringrift-ai-out.log',
      merge_logs: true,
      max_size: '100M',
      retain: 10,
    },
  ],
};
