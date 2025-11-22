import React, { useState, FormEvent } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

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
    } catch (err: any) {
      const errorData = err?.response?.data;
      const message =
        errorData?.error?.message ||
        errorData?.message ||
        err?.message ||
        'Registration failed. Please check your details and try again.';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-bold mb-2">Create an account</h1>
        <p className="text-sm text-gray-500">
          Register to play backend games, join the lobby, and track your stats. If you just want to
          explore the rules engine, you can use the local sandbox without registering.
        </p>
        {cameFromLogin && initialEmail && (
          <p className="text-xs text-emerald-400 mt-1">
            We didn’t find an account for <span className="font-mono">{initialEmail}</span>. Create
            one below.
          </p>
        )}
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
            <label htmlFor="username" className="block text-sm font-medium text-slate-100">
              Username
            </label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              placeholder="ringrift-player"
            />
          </div>

          <div className="space-y-1">
            <label htmlFor="password" className="block text-sm font-medium text-slate-100">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              placeholder="••••••••"
            />
          </div>

          <div className="space-y-1">
            <label htmlFor="confirmPassword" className="block text-sm font-medium text-slate-100">
              Confirm password
            </label>
            <input
              id="confirmPassword"
              type="password"
              autoComplete="new-password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
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
            {isSubmitting ? 'Creating account…' : 'Create account'}
          </button>

          <p className="text-xs text-slate-400">
            Already have an account?{' '}
            <Link to="/login" className="text-emerald-400 hover:text-emerald-300 underline">
              Log in
            </Link>
            .
          </p>
        </form>

        <div className="space-y-3 p-5 rounded-2xl bg-slate-900/40 border border-slate-700 shadow">
          <h2 className="text-lg font-semibold text-slate-100">Or play without registering</h2>
          <p className="text-sm text-slate-300">
            The local sandbox runs entirely in your browser. It’s ideal for exploring movement,
            captures, lines, and territory before committing to an account.
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
