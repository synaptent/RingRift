import { GameState, Move, Player } from './game';
import { User } from './user';

export interface ServerToClientEvents {
  // Game events
  'game-state': (gameState: GameState) => void;
  'move': (move: Move) => void;
  'player-joined': (player: Player) => void;
  'player-left': (playerId: string) => void;
  'game-started': (gameState: GameState) => void;
  'game-ended': (result: GameResult) => void;
  'game-paused': (reason: string) => void;
  'game-resumed': () => void;
  
  // Spectator events
  'spectator-joined': (user: User) => void;
  'spectator-left': (userId: string) => void;
  'spectator-count': (count: number) => void;
  
  // Chat events
  'chat-message': (message: ChatMessage) => void;
  'chat-history': (messages: ChatMessage[]) => void;
  
  // Time events
  'time-update': (timeInfo: TimeInfo) => void;
  'time-warning': (playerId: string, timeRemaining: number) => void;
  'time-expired': (playerId: string) => void;
  
  // Matchmaking events
  'match-found': (gameId: string) => void;
  'matchmaking-status': (status: MatchmakingStatus) => void;
  'queue-position': (position: number) => void;
  
  // System events
  'error': (error: SocketError) => void;
  'notification': (notification: Notification) => void;
  'user-status': (userId: string, status: string) => void;
  'server-message': (message: string) => void;
  
  // Connection events
  'connected': () => void;
  'disconnected': (reason: string) => void;
  'reconnected': () => void;
}

export interface ClientToServerEvents {
  // Game actions
  'join-game': (gameId: string) => void;
  'leave-game': (gameId: string) => void;
  'make-move': (move: MoveRequest) => void;
  'resign': (gameId: string) => void;
  'offer-draw': (gameId: string) => void;
  'accept-draw': (gameId: string) => void;
  'decline-draw': (gameId: string) => void;
  'request-undo': (gameId: string) => void;
  'accept-undo': (gameId: string) => void;
  'decline-undo': (gameId: string) => void;
  
  // Spectator actions
  'spectate-game': (gameId: string) => void;
  'stop-spectating': (gameId: string) => void;
  
  // Chat actions
  'send-message': (message: SendMessageRequest) => void;
  'typing-start': (gameId: string) => void;
  'typing-stop': (gameId: string) => void;
  
  // Matchmaking actions
  'join-queue': (preferences: MatchmakingPreferences) => void;
  'leave-queue': () => void;
  'create-game': (gameConfig: CreateGameRequest) => void;
  'join-private-game': (gameCode: string) => void;
  
  // User actions
  'update-status': (status: string) => void;
  'get-online-users': () => void;
  'challenge-user': (userId: string, gameConfig: CreateGameRequest) => void;
  'accept-challenge': (challengeId: string) => void;
  'decline-challenge': (challengeId: string) => void;
  
  // System actions
  'ping': () => void;
  'authenticate': (token: string) => void;
  'heartbeat': () => void;
}

export interface InterServerEvents {
  'game-update': (gameId: string, update: GameUpdate) => void;
  'user-connected': (userId: string, socketId: string) => void;
  'user-disconnected': (userId: string, socketId: string) => void;
  'broadcast-message': (message: string) => void;
}

export interface SocketData {
  userId: string;
  username: string;
  currentGameId?: string;
  isSpectating?: string[];
  lastActivity: Date;
  authenticated: boolean;
}

// Request/Response types
export interface MoveRequest {
  gameId: string;
  move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>;
}

export interface CreateGameRequest {
  boardType: string;
  timeControl: {
    initialTime: number;
    increment: number;
  };
  isRated: boolean;
  isPrivate: boolean;
  maxPlayers: number;
  aiOpponents?: {
    count: number;
    difficulty: number[];
  };
}

export interface MatchmakingPreferences {
  boardType: string;
  timeControl: {
    min: number;
    max: number;
  };
  ratingRange: {
    min: number;
    max: number;
  };
  allowAI: boolean;
}

export interface SendMessageRequest {
  gameId: string;
  content: string;
  type: 'game' | 'spectator' | 'private';
  recipientId?: string;
}

// Event data types
export interface ChatMessage {
  id: string;
  gameId: string;
  userId: string;
  username: string;
  content: string;
  type: 'game' | 'spectator' | 'system' | 'private';
  timestamp: Date;
  edited?: boolean;
  editedAt?: Date;
}

export interface TimeInfo {
  gameId: string;
  players: {
    [playerId: string]: {
      timeRemaining: number;
      isActive: boolean;
    };
  };
  lastUpdate: Date;
}

export interface GameResult {
  gameId: string;
  winner?: string;
  reason: 'rings_removed' | 'timeout' | 'resignation' | 'draw' | 'abandonment';
  finalScore: { [playerId: string]: number };
  ratingChanges?: { [playerId: string]: number };
  endedAt: Date;
}

export interface MatchmakingStatus {
  inQueue: boolean;
  estimatedWaitTime?: number;
  queuePosition?: number;
  searchCriteria: MatchmakingPreferences;
}

export interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  persistent?: boolean;
  actions?: NotificationAction[];
}

export interface NotificationAction {
  label: string;
  action: string;
  data?: any;
}

export interface SocketError {
  code: string;
  message: string;
  details?: any;
  timestamp: Date;
}

export interface GameUpdate {
  type: 'move' | 'player_joined' | 'player_left' | 'game_ended' | 'state_change';
  data: any;
  timestamp: Date;
}

// Socket event validation schemas
export const SOCKET_EVENTS = {
  // Client events that require authentication
  AUTHENTICATED_EVENTS: [
    'join-game',
    'leave-game',
    'make-move',
    'resign',
    'offer-draw',
    'send-message',
    'join-queue',
    'create-game',
    'challenge-user'
  ],
  
  // Events that can be sent without authentication
  PUBLIC_EVENTS: [
    'authenticate',
    'ping',
    'spectate-game'
  ],
  
  // Events that require game participation
  GAME_EVENTS: [
    'make-move',
    'resign',
    'offer-draw',
    'accept-draw',
    'decline-draw',
    'request-undo'
  ]
} as const;

export type AuthenticatedEvent = typeof SOCKET_EVENTS.AUTHENTICATED_EVENTS[number];
export type PublicEvent = typeof SOCKET_EVENTS.PUBLIC_EVENTS[number];
export type GameEvent = typeof SOCKET_EVENTS.GAME_EVENTS[number];