import { Link } from 'react-router-dom';

export default function LoginPage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-bold mb-2">Login</h1>
        <p className="text-sm text-gray-500">Login form coming soon. In the meantime you can jump straight into a local sandbox game.</p>
      </header>

      <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-4 space-y-3 sm:space-y-0">
        <button
          type="button"
          className="px-4 py-2 rounded bg-slate-700 text-sm text-gray-300 cursor-not-allowed"
          disabled
        >
          Login (disabled)
        </button>

        <Link
          to="/sandbox"
          className="inline-flex items-center px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold text-white"
        >
          Play Local Sandbox Game
        </Link>
      </div>
    </div>
  );
}
