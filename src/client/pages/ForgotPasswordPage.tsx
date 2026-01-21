import React, { useState, FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { InlineAlert } from '../components/ui/InlineAlert';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!email) {
      setError('Please enter your email address.');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.error?.message || 'Failed to send reset email. Please try again.');
        return;
      }

      setSuccess(true);
    } catch {
      setError('Failed to send reset email. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

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
          <p className="text-sm text-slate-400">
            Enter your email address and we'll send you a link to reset your password.
          </p>
        </header>

        <div className="max-w-md">
          <section className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 shadow-lg">
            {success ? (
              <div className="space-y-4">
                <InlineAlert variant="success">
                  If an account exists with this email, a password reset link has been sent. Please
                  check your email.
                </InlineAlert>
                <p className="text-sm text-slate-300">
                  <Link
                    to="/login"
                    className="text-emerald-400 hover:text-emerald-300 underline font-medium"
                  >
                    Back to login
                  </Link>
                </p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-1">
                  <label htmlFor="email" className="block text-sm font-medium text-slate-300">
                    Email
                  </label>
                  <Input
                    id="email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                  />
                </div>

                {error && <InlineAlert variant="error">{error}</InlineAlert>}

                <Button type="submit" fullWidth disabled={isSubmitting}>
                  {isSubmitting ? 'Sending...' : 'Send Reset Link'}
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
