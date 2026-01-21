import { SESClient, SendEmailCommand } from '@aws-sdk/client-ses';
import { logger } from './logger';
import { config } from '../config';

interface EmailOptions {
  to: string;
  subject: string;
  text: string;
  html?: string;
}

// Lazily initialized SES client
let sesClient: SESClient | null = null;

/**
 * Get or create the SES client
 */
function getSESClient(): SESClient | null {
  if (config.email.provider !== 'ses' || !config.email.ses) {
    return null;
  }

  if (!sesClient) {
    sesClient = new SESClient({
      region: config.email.ses.region,
      credentials: {
        accessKeyId: config.email.ses.accessKeyId,
        secretAccessKey: config.email.ses.secretAccessKey,
      },
    });
  }

  return sesClient;
}

/**
 * Send email via AWS SES
 */
async function sendViaSES(options: EmailOptions): Promise<boolean> {
  const client = getSESClient();
  if (!client) {
    logger.error('SES client not configured');
    return false;
  }

  const fromAddress = config.email.from || 'noreply@ringrift.ai';

  try {
    const command = new SendEmailCommand({
      Source: fromAddress,
      Destination: {
        ToAddresses: [options.to],
      },
      Message: {
        Subject: {
          Data: options.subject,
          Charset: 'UTF-8',
        },
        Body: {
          Text: {
            Data: options.text,
            Charset: 'UTF-8',
          },
          ...(options.html && {
            Html: {
              Data: options.html,
              Charset: 'UTF-8',
            },
          }),
        },
      },
    });

    const response = await client.send(command);
    logger.info('Email sent via SES', {
      to: options.to,
      subject: options.subject,
      messageId: response.MessageId,
    });
    return true;
  } catch (error) {
    logger.error('Failed to send email via SES', {
      to: options.to,
      subject: options.subject,
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

/**
 * Mock email sender for development/testing
 * Logs the email content instead of sending it
 */
async function sendViaMock(options: EmailOptions): Promise<boolean> {
  logger.info('MOCK EMAIL SENT', {
    to: options.to,
    subject: options.subject,
    text: options.text,
    html: options.html ? `${options.html.substring(0, 100)}...` : undefined,
  });

  // Simulate network delay outside of Jest tests
  if (!config.isTest) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  return true;
}

/**
 * Send an email using the configured provider
 *
 * Provider selection:
 * - 'ses': AWS Simple Email Service
 * - 'smtp': SMTP server (not yet implemented)
 * - 'mock': Log only (default for development)
 */
export const sendEmail = async (options: EmailOptions): Promise<boolean> => {
  const provider = config.email.provider;

  logger.info('sendEmail called', {
    provider,
    to: options.to,
    subject: options.subject,
    hasSesConfig: !!config.email.ses,
  });

  switch (provider) {
    case 'ses':
      return sendViaSES(options);
    case 'smtp':
      // SMTP not implemented yet - fall back to mock
      logger.warn('SMTP provider not implemented, using mock', { to: options.to });
      return sendViaMock(options);
    case 'mock':
    default:
      return sendViaMock(options);
  }
};

/**
 * Send verification email
 */
export const sendVerificationEmail = async (email: string, token: string): Promise<boolean> => {
  const verificationLink = `${config.server.publicClientUrl}/verify-email?token=${token}`;

  return sendEmail({
    to: email,
    subject: 'Verify your RingRift account',
    text: `Please verify your email by clicking the following link: ${verificationLink}`,
    html: `
      <h1>Welcome to RingRift!</h1>
      <p>Please verify your email by clicking the link below:</p>
      <a href="${verificationLink}">Verify Email</a>
      <p>If you didn't create an account, you can safely ignore this email.</p>
    `,
  });
};

/**
 * Send password reset email
 */
export const sendPasswordResetEmail = async (email: string, token: string): Promise<boolean> => {
  const resetLink = `${config.server.publicClientUrl}/reset-password?token=${token}`;

  return sendEmail({
    to: email,
    subject: 'Reset your RingRift password',
    text: `You requested a password reset. Click the following link to reset your password: ${resetLink}`,
    html: `
      <h1>Password Reset Request</h1>
      <p>You requested a password reset. Click the link below to reset your password:</p>
      <a href="${resetLink}">Reset Password</a>
      <p>If you didn't request this, you can safely ignore this email.</p>
      <p>This link will expire in 1 hour.</p>
    `,
  });
};
