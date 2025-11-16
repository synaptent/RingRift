import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';

const AI_USER_EMAIL = 'ai@ringrift.internal';
const AI_USERNAME = 'RingRift AI';
const AI_PASSWORD_PLACEHOLDER = 'AI_USER_NO_LOGIN';

/**
 * Ensure there is a dedicated User row representing the system AI.
 *
 * This allows us to persist AI moves in the Move table using a
 * legitimate playerId while keeping the Prisma schema unchanged.
 */
export async function getOrCreateAIUser() {
  const prisma = getDatabaseClient();
  if (!prisma) {
    throw new Error('Database not available');
  }

  let user = await prisma.user.findUnique({ where: { email: AI_USER_EMAIL } });
  if (user) {
    return user;
  }

  user = await prisma.user.create({
    data: {
      email: AI_USER_EMAIL,
      username: AI_USERNAME,
      passwordHash: AI_PASSWORD_PLACEHOLDER,
      role: 'USER',
      rating: 1500,
      isActive: true,
      emailVerified: true
    }
  });

  logger.info('AI system user created', { userId: user.id, email: AI_USER_EMAIL });
  return user;
}
