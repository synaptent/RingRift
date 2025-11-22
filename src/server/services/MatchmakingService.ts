import { MatchmakingPreferences, MatchmakingStatus } from '../../shared/types/websocket';
import { GameStatus } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { WebSocketServer } from '../websocket/server';
import { logger } from '../utils/logger';
import { v4 as uuidv4 } from 'uuid';

interface QueueEntry {
  userId: string;
  socketId: string;
  preferences: MatchmakingPreferences;
  rating: number;
  joinedAt: Date;
  ticketId: string;
}

export class MatchmakingService {
  private queue: QueueEntry[] = [];
  private matchCheckInterval: NodeJS.Timeout | null = null;
  private readonly MATCH_CHECK_INTERVAL_MS = 5000;
  private readonly RATING_EXPANSION_RATE = 50; // Rating range expands by this amount per interval
  private readonly MAX_WAIT_TIME_MS = 60000; // 1 minute max wait before expanding to any rating

  constructor(private wsServer: WebSocketServer) {
    this.startMatchmakingLoop();
  }

  public addToQueue(
    userId: string,
    socketId: string,
    preferences: MatchmakingPreferences,
    rating: number
  ): string {
    // Remove existing entry if present
    this.removeFromQueue(userId);

    const ticketId = uuidv4();
    const entry: QueueEntry = {
      userId,
      socketId,
      preferences,
      rating,
      joinedAt: new Date(),
      ticketId,
    };

    this.queue.push(entry);
    this.emitStatus(entry);

    logger.info('User added to matchmaking queue', { userId, rating, preferences });

    // Try to find a match immediately
    this.findMatch(entry);

    return ticketId;
  }

  public removeFromQueue(userId: string): void {
    const index = this.queue.findIndex((e) => e.userId === userId);
    if (index !== -1) {
      this.queue.splice(index, 1);
      logger.info('User removed from matchmaking queue', { userId });
    }
  }

  private startMatchmakingLoop() {
    if (this.matchCheckInterval) return;

    this.matchCheckInterval = setInterval(() => {
      this.processQueue();
    }, this.MATCH_CHECK_INTERVAL_MS);
  }

  private processQueue() {
    // Sort queue by join time (FCFS)
    this.queue.sort((a, b) => a.joinedAt.getTime() - b.joinedAt.getTime());

    // Try to match each player
    // Note: We iterate backwards or use a while loop to handle removals safely
    // but for simplicity here we just iterate and skip if already matched
    const matchedUserIds = new Set<string>();

    for (const entry of this.queue) {
      if (matchedUserIds.has(entry.userId)) continue;

      const match = this.findMatch(entry);
      if (match) {
        matchedUserIds.add(entry.userId);
        matchedUserIds.add(match.userId);
      } else {
        // Update status for unmatched players (e.g. expanded range)
        this.emitStatus(entry);
      }
    }
  }

  private findMatch(player: QueueEntry): QueueEntry | null {
    const now = Date.now();
    const waitTime = now - player.joinedAt.getTime();

    // Calculate expanded rating range based on wait time, capped by a
    // maximum window so expansion does not grow without bound.
    const cappedWait = Math.min(
      waitTime,
      this.MATCH_CHECK_INTERVAL_MS * Math.ceil(this.MAX_WAIT_TIME_MS / this.MATCH_CHECK_INTERVAL_MS)
    );
    const expansionFactor = Math.floor(cappedWait / this.MATCH_CHECK_INTERVAL_MS);
    const ratingBuffer = this.RATING_EXPANSION_RATE * expansionFactor;

    const minRating = player.preferences.ratingRange.min - ratingBuffer;
    const maxRating = player.preferences.ratingRange.max + ratingBuffer;

    // Find a compatible opponent
    const opponent = this.queue.find((other) => {
      if (other.userId === player.userId) return false;

      // Check board type compatibility
      if (other.preferences.boardType !== player.preferences.boardType) return false;

      // Check time control compatibility (simplified: exact match on type/range)
      // In a real system, we'd check if ranges overlap
      // For now, assume preferences match if board type matches

      // Check rating compatibility (bidirectional)
      const otherWaitTime = now - other.joinedAt.getTime();
      const otherExpansion = Math.floor(otherWaitTime / this.MATCH_CHECK_INTERVAL_MS);
      const otherBuffer = this.RATING_EXPANSION_RATE * otherExpansion;

      const otherMin = other.preferences.ratingRange.min - otherBuffer;
      const otherMax = other.preferences.ratingRange.max + otherBuffer;

      const playerFitsOther = player.rating >= otherMin && player.rating <= otherMax;
      const otherFitsPlayer = other.rating >= minRating && other.rating <= maxRating;

      return playerFitsOther && otherFitsPlayer;
    });

    if (opponent) {
      this.createMatch(player, opponent);
      return opponent;
    }

    return null;
  }

  private async createMatch(player1: QueueEntry, player2: QueueEntry) {
    // Remove both from queue
    this.removeFromQueue(player1.userId);
    this.removeFromQueue(player2.userId);

    try {
      const prisma = getDatabaseClient();
      if (!prisma) throw new Error('Database not available');

      // Create game in DB
      // Note: This duplicates some logic from game routes, ideally should be shared
      const game = await prisma.game.create({
        data: {
          boardType: player1.preferences.boardType as any,
          maxPlayers: 2,
          // Use player1's time control preferences as baseline (or average)
          // For simplicity, using fixed values or player1's min
          timeControl: {
            type: 'rapid', // simplified
            initialTime: player1.preferences.timeControl.min,
            increment: 0,
          },
          isRated: true,
          allowSpectators: true,
          player1Id: player1.userId,
          player2Id: player2.userId,
          status: GameStatus.active, // Start immediately
          startedAt: new Date(),
          gameState: {}, // Initial empty state
        },
        include: {
          player1: { select: { id: true, username: true, rating: true } },
          player2: { select: { id: true, username: true, rating: true } },
        },
      });

      // Notify players
      this.wsServer.sendToUser(player1.userId, 'match-found', { gameId: game.id });
      this.wsServer.sendToUser(player2.userId, 'match-found', { gameId: game.id });

      logger.info('Match created', {
        gameId: game.id,
        player1: player1.userId,
        player2: player2.userId,
      });
    } catch (err) {
      logger.error('Failed to create match', err);
      // Re-queue players? Or notify error?
      this.wsServer.sendToUser(player1.userId, 'error', { message: 'Failed to create match' });
      this.wsServer.sendToUser(player2.userId, 'error', { message: 'Failed to create match' });
    }
  }

  private emitStatus(entry: QueueEntry) {
    const now = Date.now();
    const waitTime = now - entry.joinedAt.getTime();
    const position = this.queue.indexOf(entry) + 1;

    // Simple heuristic: decrease the remaining estimated wait time as the
    // player waits longer, but never drop below a small floor to avoid
    // reporting negative or unrealistically low values.
    const baseEstimate = 30000; // 30s baseline
    const estimatedWaitTime = Math.max(5000, baseEstimate - waitTime);

    const status: MatchmakingStatus = {
      inQueue: true,
      estimatedWaitTime,
      queuePosition: position,
      searchCriteria: entry.preferences,
    };

    this.wsServer.sendToUser(entry.userId, 'matchmaking-status', status);
  }
}
