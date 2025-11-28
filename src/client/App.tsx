import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { useAuth } from './contexts/AuthContext';
import Layout from './components/Layout';
import LoadingSpinner from './components/LoadingSpinner';

// Lazy load pages for code splitting
// GamePage is the heaviest component (2100+ lines), benefits most from lazy loading
const GamePage = lazy(() => import('./pages/GamePage'));
const LobbyPage = lazy(() => import('./pages/LobbyPage'));
const ProfilePage = lazy(() => import('./pages/ProfilePage'));
const LeaderboardPage = lazy(() => import('./pages/LeaderboardPage'));

// Auth pages are lightweight and frequently first-loaded, keep synchronous
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';

// Suspense fallback component for lazy-loaded routes
function PageLoader() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <LoadingSpinner size="lg" />
    </div>
  );
}

function App() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <>
      <Toaster
        position="top-center"
        toastOptions={{
          style: {
            background: '#1e293b',
            color: '#fff',
            border: '1px solid #334155',
          },
        }}
      />
      <Suspense fallback={<PageLoader />}>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={!user ? <LoginPage /> : <Navigate to="/" />} />
          <Route path="/register" element={!user ? <RegisterPage /> : <Navigate to="/" />} />
          {/* Public sandbox route (no auth required) */}
          <Route path="/sandbox" element={<GamePage />} />
          {/* Public spectator route (read-only, reuses GamePage HUD + log) */}
          <Route path="/spectate/:gameId" element={<GamePage />} />

          {/* Protected routes */}
          <Route path="/" element={user ? <Layout /> : <Navigate to="/login" />}>
            <Route index element={<HomePage />} />
            <Route path="lobby" element={<LobbyPage />} />
            <Route path="game/:gameId" element={<GamePage />} />
            <Route path="profile" element={<ProfilePage />} />
            <Route path="leaderboard" element={<LeaderboardPage />} />
          </Route>

          {/* Catch all route */}
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Suspense>
    </>
  );
}

export default App;
