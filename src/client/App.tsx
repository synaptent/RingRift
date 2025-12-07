import { Suspense, lazy } from 'react';
import { Routes, Route, Navigate, useParams } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { useAuth } from './contexts/AuthContext';
import { AccessibilityProvider } from './contexts/AccessibilityContext';
import Layout from './components/Layout';
import LoadingSpinner from './components/LoadingSpinner';

// Lazy load pages for code splitting
const LobbyPage = lazy(() => import('./pages/LobbyPage'));
const ProfilePage = lazy(() => import('./pages/ProfilePage'));
const LeaderboardPage = lazy(() => import('./pages/LeaderboardPage'));
const BackendGameHost = lazy(() =>
  import('./pages/BackendGameHost').then((m) => ({ default: m.BackendGameHost }))
);
const SandboxGameHost = lazy(() =>
  import('./pages/SandboxGameHost').then((m) => ({ default: m.SandboxGameHost }))
);

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

type BackendGameHostRouteParams = {
  gameId: string;
};

function BackendGameHostRoute() {
  const { gameId } = useParams<BackendGameHostRouteParams>();

  if (!gameId) {
    return <Navigate to="/" replace />;
  }

  return <BackendGameHost gameId={gameId} />;
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
    <AccessibilityProvider>
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
          <Route path="/sandbox" element={<SandboxGameHost />} />
          {/* Public spectator route (read-only, backend host) */}
          <Route path="/spectate/:gameId" element={<BackendGameHostRoute />} />

          {/* Protected routes */}
          <Route path="/" element={user ? <Layout /> : <Navigate to="/login" />}>
            <Route index element={<HomePage />} />
            <Route path="lobby" element={<LobbyPage />} />
            <Route path="game/:gameId" element={<BackendGameHostRoute />} />
            <Route path="profile" element={<ProfilePage />} />
            <Route path="leaderboard" element={<LeaderboardPage />} />
          </Route>

          {/* Catch all route */}
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Suspense>
    </AccessibilityProvider>
  );
}

export default App;
