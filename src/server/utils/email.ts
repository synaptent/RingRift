import { logger } from './logger';

interface EmailOptions {
  to: string;
  subject: string;
  text: string;
  html?: string;
}

/**
 * Mock email sender service
 * In a real application, this would use a service like SendGrid, AWS SES, or Nodemailer
 */
export const sendEmail = async (options: EmailOptions): Promise<boolean> => {
  // Log the email content instead of sending it
  logger.info('MOCK EMAIL SENT', {
    to: options.to,
    subject: options.subject,
    text: options.text,
    // Truncate HTML if it's too long for logs
    html: options.html ? `${options.html.substring(0, 100)}...` : undefined,
  });

  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 100));

  return true;
};

/**
 * Send verification email
 */
export const sendVerificationEmail = async (email: string, token: string): Promise<boolean> => {
  const verificationLink = `${process.env.CLIENT_URL || 'http://localhost:3000'}/verify-email?token=${token}`;

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
  const resetLink = `${process.env.CLIENT_URL || 'http://localhost:3000'}/reset-password?token=${token}`;

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
