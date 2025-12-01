/**
 * useReplayService - React hook for accessing the GameReplayDB API.
 *
 * Provides convenient React hooks for fetching replay data with
 * loading states, error handling, and caching via React Query.
 *
 * Usage:
 *   const { data: games, isLoading, error } = useGameList({ board_type: 'square8' });
 *   const { data: state } = useReplayState(gameId, moveNumber);
 */

import { useQuery, useMutation, useQueryClient, UseQueryResult } from '@tanstack/react-query';
import { getReplayService } from '../services/ReplayService';
import type {
  ReplayGameListResponse,
  ReplayGameMetadata,
  ReplayGameQueryParams,
  ReplayMovesResponse,
  ReplayStateResponse,
  ReplayChoicesResponse,
  ReplayStatsResponse,
  StoreGameRequest,
  StoreGameResponse,
} from '../types/replay';

// Query key factory for consistent cache keys
const replayKeys = {
  all: ['replay'] as const,
  games: () => [...replayKeys.all, 'games'] as const,
  gameList: (filters: ReplayGameQueryParams) => [...replayKeys.games(), 'list', filters] as const,
  game: (gameId: string) => [...replayKeys.games(), gameId] as const,
  state: (gameId: string, moveNumber: number) =>
    [...replayKeys.game(gameId), 'state', moveNumber] as const,
  moves: (gameId: string, start?: number, end?: number) =>
    [...replayKeys.game(gameId), 'moves', { start, end }] as const,
  choices: (gameId: string, moveNumber: number) =>
    [...replayKeys.game(gameId), 'choices', moveNumber] as const,
  stats: () => [...replayKeys.all, 'stats'] as const,
  availability: () => [...replayKeys.all, 'available'] as const,
};

/**
 * Hook to check if the replay service is available.
 */
export function useReplayServiceAvailable(): UseQueryResult<boolean, Error> {
  const service = getReplayService();

  return useQuery({
    queryKey: replayKeys.availability(),
    queryFn: () => service.isAvailable(),
    staleTime: 30_000, // Check every 30 seconds
    retry: false,
  });
}

/**
 * Hook to get database statistics.
 */
export function useReplayStats(): UseQueryResult<ReplayStatsResponse, Error> {
  const service = getReplayService();

  return useQuery({
    queryKey: replayKeys.stats(),
    queryFn: () => service.getStats(),
    staleTime: 60_000, // Stats don't change frequently
  });
}

/**
 * Hook to list games with filters.
 */
export function useGameList(
  filters: ReplayGameQueryParams = {},
  enabled = true
): UseQueryResult<ReplayGameListResponse, Error> {
  const service = getReplayService();

  return useQuery({
    queryKey: replayKeys.gameList(filters),
    queryFn: () => service.listGames(filters),
    enabled,
    staleTime: 10_000, // 10 second cache
  });
}

/**
 * Hook to get a single game with player details.
 */
export function useGame(
  gameId: string | null,
  enabled = true
): UseQueryResult<ReplayGameMetadata, Error> {
  const service = getReplayService();

  return useQuery({
    queryKey: replayKeys.game(gameId ?? ''),
    queryFn: () => {
      if (!gameId) throw new Error('Game ID required');
      return service.getGame(gameId);
    },
    enabled: enabled && gameId !== null,
    staleTime: 60_000, // Game metadata doesn't change after completion
  });
}

/**
 * Hook to get game state at a specific move.
 */
export function useReplayStateAt(
  gameId: string | null,
  moveNumber: number,
  enabled = true
): UseQueryResult<ReplayStateResponse, Error> {
  const service = getReplayService();

  return useQuery({
    queryKey: replayKeys.state(gameId ?? '', moveNumber),
    queryFn: () => {
      if (!gameId) throw new Error('Game ID required');
      return service.getStateAtMove(gameId, moveNumber);
    },
    enabled: enabled && gameId !== null,
    staleTime: Infinity, // State at a specific move never changes
  });
}

/**
 * Hook to get moves in a range.
 */
export function useMoves(
  gameId: string | null,
  start = 0,
  end?: number,
  enabled = true
): UseQueryResult<ReplayMovesResponse, Error> {
  const service = getReplayService();

  return useQuery({
    queryKey: replayKeys.moves(gameId ?? '', start, end),
    queryFn: () => {
      if (!gameId) throw new Error('Game ID required');
      return service.getMoves(gameId, start, end);
    },
    enabled: enabled && gameId !== null,
    staleTime: Infinity, // Move history never changes for completed games
  });
}

/**
 * Hook to get all moves for a game.
 * Fetches in batches and combines results.
 */
export function useAllMoves(
  gameId: string | null,
  totalMoves: number,
  enabled = true
): UseQueryResult<ReplayMovesResponse, Error> {
  const service = getReplayService();

  return useQuery({
    queryKey: [...replayKeys.game(gameId ?? ''), 'allMoves'],
    queryFn: async () => {
      if (!gameId) throw new Error('No game ID');

      // Fetch all moves in one request (use high limit)
      return service.getMoves(gameId, 0, undefined, Math.max(totalMoves, 1000));
    },
    enabled: enabled && gameId !== null && totalMoves > 0,
    staleTime: Infinity,
  });
}

/**
 * Hook to get choices at a specific move.
 */
export function useChoices(
  gameId: string | null,
  moveNumber: number,
  enabled = true
): UseQueryResult<ReplayChoicesResponse, Error> {
  const service = getReplayService();

  return useQuery({
    queryKey: replayKeys.choices(gameId ?? '', moveNumber),
    queryFn: () => {
      if (!gameId) throw new Error('Game ID required');
      return service.getChoices(gameId, moveNumber);
    },
    enabled: enabled && gameId !== null,
    staleTime: Infinity,
  });
}

/**
 * Hook to store a game (mutation).
 */
export function useStoreGame() {
  const service = getReplayService();
  const queryClient = useQueryClient();

  return useMutation<StoreGameResponse, Error, StoreGameRequest>({
    mutationFn: (request) => service.storeGame(request),
    onSuccess: () => {
      // Invalidate game list to show new game
      queryClient.invalidateQueries({ queryKey: replayKeys.games() });
      queryClient.invalidateQueries({ queryKey: replayKeys.stats() });
    },
  });
}

/**
 * Hook to prefetch game data.
 * Useful for prefetching when hovering over a game in the list.
 */
export function usePrefetchGame() {
  const service = getReplayService();
  const queryClient = useQueryClient();

  return (gameId: string) => {
    queryClient.prefetchQuery({
      queryKey: replayKeys.game(gameId),
      queryFn: () => service.getGame(gameId),
      staleTime: 60_000,
    });
  };
}

/**
 * Hook to prefetch state at move.
 * Useful for prefetching adjacent moves during playback.
 */
export function usePrefetchState() {
  const service = getReplayService();
  const queryClient = useQueryClient();

  return (gameId: string, moveNumber: number) => {
    queryClient.prefetchQuery({
      queryKey: replayKeys.state(gameId, moveNumber),
      queryFn: () => service.getStateAtMove(gameId, moveNumber),
      staleTime: Infinity,
    });
  };
}
