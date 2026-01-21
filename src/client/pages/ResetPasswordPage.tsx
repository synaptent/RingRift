import React, { useState, FormEvent } from 'react';
import { Link, useSearchParams, useNavigate } from 'react-router-dom';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { InlineAlert } from '../components/ui/InlineAlert';

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get('token');

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!password || !confirmPassword) {
      setError('Please fill in all fields.');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    if (!token) {
      setError('Invalid or missing reset token. Please request a new password reset link.');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token, newPassword: password }),
      });

      const data = await response.json();

      if (!response.ok) {
        const errorCode = data.error?.code;
        const errorMessages: Record<string, string> = {
          INVALID_TOKEN: 'This reset link has expired or is invalid. Please request a new one.',
          TOKEN_EXPIRED: 'This reset link has expired. Please request a new one.',
          WEAK_PASSWORD: 'Password must be at least 8 characters long.',
          USER_NOT_FOUND: 'Account not found. The reset link may be invalid.',
          RATE_LIMIT_EXCEEDED: 'Too many attempts. Please wait before trying again.',
        };
        setError(
          (errorCode && errorMessages[errorCode]) ||
            data.error?.message ||
            'Failed to reset password. Please try again.'
        );
        return;
      }

      setSuccess(true);
      // Redirect to login after 3 seconds
      setTimeout(() => {
        navigate('/login');
      }, 3000);
    } catch {
      setError('Failed to reset password. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!token) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100">
        <div className="container mx-auto px-2 sm:px-4 py-4 sm:py-8 space-y-4 sm:space-y-6">
          <header className="flex flex-col gap-2 sm:gap-3">
            <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold mb-1 flex items-center gap-2">
              <img
                src="/ringrift-icon.png"
                alt="RingRift"
                className="w-6 h-6 sm:w-8 sm:h-8 flex-shrink-0"
              />
              <span>RingRift – Reset Password</span>
            </h1>
          </header>

          <div className="max-w-md">
            <section className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 shadow-lg space-y-4">
              <InlineAlert variant="error">
                Invalid or missing reset token. Please request a new password reset link.
              </InlineAlert>
              <p className="text-sm text-slate-300">
                <Link
                  to="/forgot-password"
                  className="text-emerald-400 hover:text-emerald-300 underline font-medium"
                >
                  Request new reset link
                </Link>
              </p>
            </section>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="container mx-auto px-2 sm:px-4 py-4 sm:py-8 space-y-4 sm:space-y-6">
        <header className="flex flex-col gap-2 sm:gap-3">
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold mb-1 flex items-center gap-2">
            <img
              src="/ringrift-icon.png"
              alt="RingRift"
              className="w-6 h-6 sm:w-8 sm:h-8 flex-shrink-0"
            />
            <span>RingRift – Reset Password</span>
          </h1>
          <p className="text-sm text-slate-400">Enter your new password below.</p>
        </header>

        <div className="max-w-md">
          <section className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 shadow-lg">
            {success ? (
              <div className="space-y-4">
                <InlineAlert variant="success">
                  Password reset successfully! Redirecting to login...
                </InlineAlert>
                <p className="text-sm text-slate-300">
                  <Link
                    to="/login"
                    className="text-emerald-400 hover:text-emerald-300 underline font-medium"
                  >
                    Click here if not redirected
                  </Link>
                </p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-1">
                  <label htmlFor="password" className="block text-sm font-medium text-slate-300">
                    New Password
                  </label>
                  <Input
                    id="password"
                    type="password"
                    autoComplete="new-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="At least 8 characters"
                  />
                </div>

                <div className="space-y-1">
                  <label
                    htmlFor="confirmPassword"
                    className="block text-sm font-medium text-slate-300"
                  >
                    Confirm New Password
                  </label>
                  <Input
                    id="confirmPassword"
                    type="password"
                    autoComplete="new-password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm your password"
                  />
                </div>

                {error && <InlineAlert variant="error">{error}</InlineAlert>}

                <Button type="submit" fullWidth disabled={isSubmitting}>
                  {isSubmitting ? 'Resetting...' : 'Reset Password'}
                </Button>

                <div className="pt-3 border-t border-slate-700/50">
                  <p className="text-sm text-slate-300">
                    Remember your password?{' '}
                    <Link
                      to="/login"
                      className="text-emerald-400 hover:text-emerald-300 underline font-medium"
                    >
                      Sign in
                    </Link>
                  </p>
                </div>
              </form>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
