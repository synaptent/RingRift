import React, { useState, FormEvent } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { InlineAlert } from '../components/ui/InlineAlert';
import { extractErrorMessage, extractErrorCode } from '../utils/errorReporting';

export default function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const locationState = location.state as { email?: string } | null;
  const initialEmail = locationState?.email ?? '';

  const [email, setEmail] = useState(initialEmail);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const cameFromLogin = Boolean(initialEmail);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!email || !username || !password || !confirmPassword) {
      setError('Please fill in all fields.');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await register(email, username, password, confirmPassword);
      // On successful registration, you are logged in via AuthContext
      // and redirected into the main app shell.
      navigate('/');
    } catch (err: unknown) {
      const errorCode = extractErrorCode(err);
      const serverMessage = extractErrorMessage(err, '');

      // Map error codes to user-friendly messages
      const errorMessages: Record<string, string> = {
        EMAIL_ALREADY_EXISTS: 'An account with this email already exists. Please login instead.',
        USERNAME_ALREADY_EXISTS: 'This username is already taken. Please choose another.',
        WEAK_PASSWORD: 'Password must be at least 8 characters long.',
        INVALID_EMAIL: 'Please enter a valid email address.',
        INVALID_USERNAME: 'Username can only contain letters, numbers, and underscores.',
        RATE_LIMIT_EXCEEDED: 'Too many attempts. Please wait before trying again.',
        PASSWORDS_DO_NOT_MATCH: 'Passwords do not match.',
      };

      const displayMessage =
        (errorCode && errorMessages[errorCode]) ||
        serverMessage ||
        'Registration failed. Please check your details and try again.';

      setError(displayMessage);
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
            <span>RingRift – Create Account</span>
          </h1>
          <p className="text-sm text-slate-400">
            Register to play backend games, join the lobby, and track your stats. If you just want
            to explore the rules engine, you can use the local sandbox without registering.
          </p>
          {cameFromLogin && initialEmail && (
            <p className="text-xs text-emerald-400">
              We didn't find an account for <span className="font-mono">{initialEmail}</span>.
              Create one below.
            </p>
          )}
        </header>

        <div className="grid gap-4 sm:gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <section className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 shadow-lg">
            <div className="mb-4">
              <p className="text-xs uppercase tracking-wide text-slate-400">Account</p>
              <h2 className="text-lg font-semibold text-white">Create your account</h2>
            </div>

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

              <div className="space-y-1">
                <label htmlFor="username" className="block text-sm font-medium text-slate-300">
                  Username
                </label>
                <Input
                  id="username"
                  type="text"
                  autoComplete="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="ringrift-player"
                />
              </div>

              <div className="space-y-1">
                <label htmlFor="password" className="block text-sm font-medium text-slate-300">
                  Password
                </label>
                <Input
                  id="password"
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                />
                <p className="text-xs text-slate-500">
                  At least 8 characters with uppercase, lowercase, and a number
                </p>
              </div>

              <div className="space-y-1">
                <label
                  htmlFor="confirmPassword"
                  className="block text-sm font-medium text-slate-300"
                >
                  Confirm password
                </label>
                <Input
                  id="confirmPassword"
                  type="password"
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                />
              </div>

              {error && <InlineAlert variant="error">{error}</InlineAlert>}

              <Button type="submit" fullWidth disabled={isSubmitting}>
                {isSubmitting ? 'Creating account…' : 'Create account'}
              </Button>

              <div className="pt-3 border-t border-slate-700/50">
                <p className="text-sm text-slate-300">
                  Already have an account?{' '}
                  <Link
                    to="/login"
                    className="text-emerald-400 hover:text-emerald-300 underline font-medium"
                  >
                    Log in
                  </Link>
                </p>
              </div>
            </form>
          </section>

          <section className="p-5 rounded-2xl bg-slate-900/70 border border-slate-700 shadow-lg space-y-4">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-400">Quick Start</p>
              <h2 className="text-lg font-semibold text-white">Play without registering</h2>
            </div>

            <p className="text-sm text-slate-300">
              The local sandbox runs entirely in your browser. It's ideal for exploring movement,
              captures, lines, and territory before committing to an account.
            </p>

            <div className="space-y-3">
              <Link
                to="/sandbox"
                className="block w-full px-4 py-2.5 rounded-xl border border-slate-600 bg-slate-900/60 text-center text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition text-sm font-medium"
              >
                Play Local Sandbox Game
              </Link>

              <div className="flex flex-wrap gap-2">
                <Link
                  to="/sandbox?preset=learn-basics"
                  className="flex-1 px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-center text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition text-sm font-medium"
                >
                  Learn the Basics
                </Link>
                <Link
                  to="/help"
                  className="flex-1 px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-center text-slate-200 hover:border-sky-400 hover:text-sky-200 transition text-sm font-medium"
                >
                  Help Topics
                </Link>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
