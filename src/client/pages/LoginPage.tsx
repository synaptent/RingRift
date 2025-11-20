import React, { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await login(email, password);
      // On successful login, redirect to the main app shell (home/lobby).
      navigate('/');
    } catch (err: any) {
      const errorData = err?.response?.data;
      const errorCode = errorData?.error?.code as string | undefined;
      const backendMessage =
        errorData?.error?.message ||
        errorData?.message ||
        err?.message;

      // If the backend reports invalid credentials for this email, assume this may
      // be a new user and send them directly to the registration flow, carrying
      // the email along so it can be pre-filled.
      if (errorCode === 'INVALID_CREDENTIALS') {
        navigate('/register', { state: { email } });
        return;
      }

      setError(
        backendMessage ||
          'Login failed. Please check your credentials and try again.'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-bold mb-2">Login</h1>
        <p className="text-sm text-gray-500">
          Sign in to play backend games, use the lobby, and track your stats. You can also jump
          straight into a local sandbox game without an account.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-[minmax(0,2fr)_minmax(0,1.2fr)]">
        <form
          onSubmit={handleSubmit}
          className="space-y-4 p-5 rounded-2xl bg-slate-900/70 border border-slate-700 shadow-lg"
        >
          <div className="space-y-1">
            <label htmlFor="email" className="block text-sm font-medium text-slate-100">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              placeholder="you@example.com"
            />
          </div>

          <div className="space-y-1">
            <label htmlFor="password" className="block text-sm font-medium text-slate-100">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <div className="text-sm text-red-300 bg-red-900/40 border border-red-700 rounded-md px-3 py-2">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="w-full inline-flex items-center justify-center px-4 py-2 rounded-md bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold text-white disabled:opacity-60 disabled:cursor-not-allowed"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Logging in…' : 'Login'}
          </button>

          <p className="text-xs text-slate-400">
            Don't have an account yet?{' '}
            <Link
              to="/register"
              className="text-emerald-400 hover:text-emerald-300 underline"
            >
              Create an account
            </Link>
            .
          </p>
        </form>

        <div className="space-y-3 p-5 rounded-2xl bg-slate-900/40 border border-slate-700 shadow">
          <h2 className="text-lg font-semibold text-slate-100">Or play without logging in</h2>
          <p className="text-sm text-slate-300">
            The local sandbox runs the full rules engine entirely in your browser. It's ideal
            for experimenting with movement, captures, lines, and territory without creating an
            account.
          </p>
          <Link
            to="/sandbox"
            className="inline-flex items-center justify-center px-4 py-2 rounded-md bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold text-white"
          >
            Play Local Sandbox Game
          </Link>
        </div>
      </div>
    </div>
  );
}
