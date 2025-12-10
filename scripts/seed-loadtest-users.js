/* Seed load-test users directly via Prisma.
 *
 * Environment variables:
 *   LOADTEST_USER_COUNT    - Number of users to create (default: 400)
 *   LOADTEST_USER_PASSWORD - Password for all users (default: TestPassword123!)
 *   LOADTEST_USER_OFFSET   - Starting index for user numbering (default: 1)
 *   LOADTEST_USER_DOMAIN   - Email domain (default: loadtest.local)
 */
'use strict';

const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcryptjs');

const prisma = new PrismaClient();

async function main() {
  const userCount = parseInt(process.env.LOADTEST_USER_COUNT || '400', 10);
  const startOffset = parseInt(process.env.LOADTEST_USER_OFFSET || '1', 10);
  const domain = process.env.LOADTEST_USER_DOMAIN || 'loadtest.local';
  const passwordPlain = process.env.LOADTEST_USER_PASSWORD || 'TestPassword123!';
  const saltRounds = 12;
  const passwordHash = await bcrypt.hash(passwordPlain, saltRounds);

  console.log(`Seeding ${userCount} load-test users (offset=${startOffset}, domain=${domain})...`);

  const users = Array.from({ length: userCount }, (_, idx) => {
    const i = startOffset + idx;
    return {
      email: `loadtest_user_${i}@${domain}`,
      username: `loadtest_user_${i}`,
    };
  });

  let created = 0;
  let skipped = 0;

  for (const user of users) {
    const existing = await prisma.user.findUnique({
      where: { email: user.email },
    });

    if (existing) {
      skipped++;
      continue;
    }

    await prisma.user.create({
      data: {
        email: user.email,
        username: user.username,
        passwordHash,
      },
    });
    created++;
    if (created % 50 === 0) {
      console.log(`  Created ${created} users...`);
    }
  }

  console.log(`Done: ${created} created, ${skipped} skipped (already exist)`);
}

main()
  .catch((err) => {
    console.error('Error seeding load-test users:', err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });