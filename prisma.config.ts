import path from 'node:path';
import { defineConfig } from 'prisma/config';

const databaseUrl = process.env.DATABASE_URL;

export default defineConfig({
  earlyAccess: true,
  schema: path.join(__dirname, 'prisma', 'schema.prisma'),
  ...(databaseUrl ? { datasource: { url: databaseUrl } } : {}),
});
