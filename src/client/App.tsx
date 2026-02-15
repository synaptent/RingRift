import { Suspense, lazy, useEffect, useRef } from 'react';
import { Routes, Route, Navigate, useParams, useNavigate, useLocation } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { useAuth } from './contexts/AuthContext';
import { AccessibilityProvider } from './contexts/AccessibilityContext';
import { SoundProvider } from './contexts/SoundContext';
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
const HelpPage = lazy(() => import('./pages/HelpPage'));
const JoinByInvitePage = lazy(() => import('./pages/JoinByInvitePage'));
const GameHistoryPage = lazy(() => import('./pages/GameHistoryPage'));
const PlayerProfilePage = lazy(() => import('./pages/PlayerProfilePage'));

// Auth pages are lightweight and frequently first-loaded, keep synchronous
import HomePage from './pages/HomePage';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ForgotPasswordPage from './pages/ForgotPasswordPage';
import ResetPasswordPage from './pages/ResetPasswordPage';
import NotFoundPage from './pages/NotFoundPage';

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

/**
 * ProtectedRoute wrapper that handles auth transitions gracefully.
 *
 * Instead of immediately switching from Layout to Navigate (which can cause
 * React hooks ordering issues when auth state changes mid-render), this
 * component defers the redirect to a useEffect, ensuring clean unmounting.
 */
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Track whether user was authenticated when component mounted.
  // This prevents redirect loops and handles the transition gracefully.
  const wasAuthenticatedRef = useRef(!!user);

  useEffect(() => {
    if (!user) {
      // User became unauthenticated - redirect after a tick to ensure
      // clean unmounting of child components
      const timer = setTimeout(() => {
        navigate('/login', {
          replace: true,
          state: { from: location.pathname },
        });
      }, 0);
      return () => clearTimeout(timer);
    }
    wasAuthenticatedRef.current = true;
    return; // Explicit return for TypeScript
  }, [user, navigate, location.pathname]);

  // If user just became null, render nothing for one frame to allow
  // clean unmounting before the redirect
  if (!user && wasAuthenticatedRef.current) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  // If never authenticated, don't render protected content
  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return <>{children}</>;
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
      <SoundProvider>
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
            {/* Public landing page for unauthenticated visitors */}
            {!user && <Route path="/" element={<LandingPage />} />}

            {/* Public routes */}
            <Route path="/login" element={!user ? <LoginPage /> : <Navigate to="/" />} />
            <Route path="/register" element={!user ? <RegisterPage /> : <Navigate to="/" />} />
            <Route
              path="/forgot-password"
              element={!user ? <ForgotPasswordPage /> : <Navigate to="/" />}
            />
            <Route
              path="/reset-password"
              element={!user ? <ResetPasswordPage /> : <Navigate to="/" />}
            />
            {/* Public sandbox route (no auth required) */}
            <Route path="/sandbox" element={<SandboxGameHost />} />
            <Route path="/help" element={<HelpPage />} />
            <Route path="/help/:topic" element={<HelpPage />} />
            {/* Public invite link - resolves code and joins game */}
            <Route path="/join/:inviteCode" element={<JoinByInvitePage />} />
            {/* Public spectator route (read-only, backend host) */}
            <Route path="/spectate/:gameId" element={<BackendGameHostRoute />} />

            {/* Protected routes - wrapped in ProtectedRoute for graceful auth transition */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <Layout />
                </ProtectedRoute>
              }
            >
              <Route index element={<HomePage />} />
              <Route path="lobby" element={<LobbyPage />} />
              <Route path="game/:gameId" element={<BackendGameHostRoute />} />
              <Route path="profile" element={<ProfilePage />} />
              <Route path="history" element={<GameHistoryPage />} />
              <Route path="leaderboard" element={<LeaderboardPage />} />
              <Route path="player/:userId" element={<PlayerProfilePage />} />
            </Route>

            {/* Catch all route */}
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Suspense>
      </SoundProvider>
    </AccessibilityProvider>
  );
}

export default App;
