#!/usr/bin/env node
/**
 * Load Test User Pool Seeding Script
 *
 * Seeds the database with load test users for k6 scenarios.
 * Users are created with emails like loadtest_user_1@loadtest.local
 *
 * Usage:
 *   node tests/load/scripts/seed-users.js [options]
 *
 * Options:
 *   --count N          Number of users to seed (default: 100)
 *   --base-url URL     Backend API URL (default: http://localhost:3000)
 *   --password PASS    Password for all users (default: LoadTestK6Pass123)
 *   --prefix PREFIX    Email prefix (default: loadtest_user_)
 *   --domain DOMAIN    Email domain (default: loadtest.local)
 *   --dry-run          Show what would be created without making requests
 *   --verbose          Show detailed output
 *
 * Environment Variables:
 *   BASE_URL                    - Override default base URL
 *   LOADTEST_USER_POOL_PASSWORD - Override default password
 */

const https = require('https');
const http = require('http');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  count: 100,
  baseUrl: process.env.BASE_URL || 'http://localhost:3000',
  password: process.env.LOADTEST_USER_POOL_PASSWORD || 'LoadTestK6Pass123',
  prefix: 'loadtest_user_',
  domain: 'loadtest.local',
  dryRun: false,
  verbose: false,
};

for (let i = 0; i < args.length; i++) {
  switch (args[i]) {
    case '--count':
      options.count = parseInt(args[++i], 10);
      break;
    case '--base-url':
      options.baseUrl = args[++i];
      break;
    case '--password':
      options.password = args[++i];
      break;
    case '--prefix':
      options.prefix = args[++i];
      break;
    case '--domain':
      options.domain = args[++i];
      break;
    case '--dry-run':
      options.dryRun = true;
      break;
    case '--verbose':
      options.verbose = true;
      break;
    case '--help':
      console.log(`
Load Test User Pool Seeding Script

Usage: node seed-users.js [options]

Options:
  --count N          Number of users to seed (default: 100)
  --base-url URL     Backend API URL (default: http://localhost:3000)
  --password PASS    Password for all users (default: LoadTestK6Pass123)
  --prefix PREFIX    Email prefix (default: loadtest_user_)
  --domain DOMAIN    Email domain (default: loadtest.local)
  --dry-run          Show what would be created without making requests
  --verbose          Show detailed output
  --help             Show this help message
`);
      process.exit(0);
  }
}

// Validate password meets requirements (lowercase, uppercase, number)
if (!/[a-z]/.test(options.password) || !/[A-Z]/.test(options.password) || !/[0-9]/.test(options.password)) {
  console.error('Error: Password must contain lowercase, uppercase, and number');
  process.exit(1);
}

console.log(`
╔════════════════════════════════════════════════════════════════╗
║           Load Test User Pool Seeding                          ║
╠════════════════════════════════════════════════════════════════╣
║ Base URL:  ${options.baseUrl.padEnd(48)}║
║ Users:     ${String(options.count).padEnd(48)}║
║ Pattern:   ${(options.prefix + 'N@' + options.domain).padEnd(48)}║
║ Dry Run:   ${String(options.dryRun).padEnd(48)}║
╚════════════════════════════════════════════════════════════════╝
`);

if (options.dryRun) {
  console.log('DRY RUN - No users will be created\n');
  console.log('Would create the following users:');
  for (let i = 1; i <= Math.min(options.count, 5); i++) {
    console.log(`  ${options.prefix}${i}@${options.domain}`);
  }
  if (options.count > 5) {
    console.log(`  ... and ${options.count - 5} more`);
  }
  process.exit(0);
}

// Helper to make HTTP requests
function makeRequest(url, method, data) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);
    const isHttps = parsedUrl.protocol === 'https:';
    const lib = isHttps ? https : http;

    const requestOptions = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (isHttps ? 443 : 80),
      path: parsedUrl.pathname,
      method: method,
      headers: {
        'Content-Type': 'application/json',
      },
      // Skip SSL verification for staging/dev
      rejectUnauthorized: false,
    };

    const req = lib.request(requestOptions, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          const json = JSON.parse(body);
          resolve({ status: res.statusCode, body: json });
        } catch {
          resolve({ status: res.statusCode, body: body });
        }
      });
    });

    req.on('error', reject);

    if (data) {
      req.write(JSON.stringify(data));
    }
    req.end();
  });
}

// Create a single user
async function createUser(index) {
  const email = `${options.prefix}${index}@${options.domain}`;
  const username = `${options.prefix}${index}`;

  try {
    const response = await makeRequest(
      `${options.baseUrl}/api/auth/register`,
      'POST',
      {
        email,
        username,
        password: options.password,
        confirmPassword: options.password,
      }
    );

    if (response.status === 201 || response.status === 200) {
      return { success: true, email, status: 'created' };
    } else if (response.status === 409) {
      return { success: true, email, status: 'exists' };
    } else {
      return {
        success: false,
        email,
        status: 'error',
        error: response.body?.error?.message || `HTTP ${response.status}`
      };
    }
  } catch (error) {
    return { success: false, email, status: 'error', error: error.message };
  }
}

// Main seeding function
async function seedUsers() {
  const results = { created: 0, exists: 0, errors: 0 };
  const startTime = Date.now();

  // Process in batches to avoid overwhelming the server
  const batchSize = 10;
  const batches = Math.ceil(options.count / batchSize);

  for (let batch = 0; batch < batches; batch++) {
    const batchStart = batch * batchSize + 1;
    const batchEnd = Math.min((batch + 1) * batchSize, options.count);

    const promises = [];
    for (let i = batchStart; i <= batchEnd; i++) {
      promises.push(createUser(i));
    }

    const batchResults = await Promise.all(promises);

    for (const result of batchResults) {
      if (result.status === 'created') {
        results.created++;
        if (options.verbose) console.log(`✓ Created: ${result.email}`);
      } else if (result.status === 'exists') {
        results.exists++;
        if (options.verbose) console.log(`○ Exists:  ${result.email}`);
      } else {
        results.errors++;
        console.error(`✗ Error:   ${result.email} - ${result.error}`);
      }
    }

    // Progress update
    const progress = Math.round((batchEnd / options.count) * 100);
    process.stdout.write(`\rProgress: ${progress}% (${batchEnd}/${options.count})`);
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  console.log(`\n
╔════════════════════════════════════════════════════════════════╗
║                     Seeding Complete                            ║
╠════════════════════════════════════════════════════════════════╣
║ Created:   ${String(results.created).padEnd(48)}║
║ Existed:   ${String(results.exists).padEnd(48)}║
║ Errors:    ${String(results.errors).padEnd(48)}║
║ Time:      ${(elapsed + 's').padEnd(48)}║
╚════════════════════════════════════════════════════════════════╝
`);

  if (results.errors > 0) {
    process.exit(1);
  }
}

// Run the seeding
seedUsers().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
