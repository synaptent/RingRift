/**
 * RematchService Unit Tests
 *
 * Tests for the rematch request system including:
 * - Creating rematch requests
 * - Accepting/declining rematch requests
 * - Request expiration
 * - Validation rules
 */

import { RematchService, getRematchService } from '../../src/server/services/RematchService';

// Mock dependencies
const mockPrisma = {
  rematchRequest: {
    findFirst: jest.fn(),
    findUnique: jest.fn(),
    create: jest.fn(),
    update: jest.fn(),
    updateMany: jest.fn(),
  },
  game: {
    findUnique: jest.fn(),
  },
};

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => mockPrisma),
}));

jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('RematchService', () => {
  let service: RematchService;

  const mockGame = {
    id: 'game-1',
    status: 'completed',
    player1Id: 'user-1',
    player2Id: 'user-2',
    player3Id: null,
    player4Id: null,
  };

  const mockRequest = {
    id: 'request-1',
    gameId: 'game-1',
    requesterId: 'user-1',
    status: 'pending',
    expiresAt: new Date(Date.now() + 30000),
    createdAt: new Date(),
    respondedAt: null,
    newGameId: null,
    requester: { username: 'User1' },
    game: mockGame,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    service = new RematchService();
  });

  describe('createRematchRequest', () => {
    it('should create a rematch request successfully', async () => {
      mockPrisma.rematchRequest.findFirst.mockResolvedValue(null);
      mockPrisma.game.findUnique.mockResolvedValue(mockGame);
      mockPrisma.rematchRequest.create.mockResolvedValue(mockRequest);

      const result = await service.createRematchRequest('game-1', 'user-1');

      expect(result.success).toBe(true);
      expect(result.request).toBeDefined();
      expect(result.request?.gameId).toBe('game-1');
      expect(result.request?.requesterId).toBe('user-1');
    });

    it('should reject if pending request already exists', async () => {
      mockPrisma.rematchRequest.findFirst.mockResolvedValue(mockRequest);

      const result = await service.createRematchRequest('game-1', 'user-1');

      expect(result.success).toBe(false);
      expect(result.error).toContain('already pending');
    });

    it('should reject if game not found', async () => {
      mockPrisma.rematchRequest.findFirst.mockResolvedValue(null);
      mockPrisma.game.findUnique.mockResolvedValue(null);

      const result = await service.createRematchRequest('game-1', 'user-1');

      expect(result.success).toBe(false);
      expect(result.error).toBe('Game not found');
    });

    it('should reject if game is not completed', async () => {
      mockPrisma.rematchRequest.findFirst.mockResolvedValue(null);
      mockPrisma.game.findUnique.mockResolvedValue({ ...mockGame, status: 'active' });

      const result = await service.createRematchRequest('game-1', 'user-1');

      expect(result.success).toBe(false);
      expect(result.error).toContain('completed games');
    });

    it('should reject if requester was not a player', async () => {
      mockPrisma.rematchRequest.findFirst.mockResolvedValue(null);
      mockPrisma.game.findUnique.mockResolvedValue(mockGame);

      const result = await service.createRematchRequest('game-1', 'user-3');

      expect(result.success).toBe(false);
      expect(result.error).toContain('Only players');
    });

    it('should accept finished status as completed', async () => {
      mockPrisma.rematchRequest.findFirst.mockResolvedValue(null);
      mockPrisma.game.findUnique.mockResolvedValue({ ...mockGame, status: 'finished' });
      mockPrisma.rematchRequest.create.mockResolvedValue(mockRequest);

      const result = await service.createRematchRequest('game-1', 'user-1');

      expect(result.success).toBe(true);
    });
  });

  describe('acceptRematch', () => {
    const createGameFn = jest.fn().mockResolvedValue('new-game-id');

    beforeEach(() => {
      createGameFn.mockClear();
    });

    it('should accept rematch and create new game', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue(mockRequest);
      mockPrisma.rematchRequest.update.mockResolvedValue({
        ...mockRequest,
        status: 'accepted',
        newGameId: 'new-game-id',
      });

      const result = await service.acceptRematch('request-1', 'user-2', createGameFn);

      expect(result.success).toBe(true);
      expect(result.newGameId).toBe('new-game-id');
      expect(createGameFn).toHaveBeenCalledWith('game-1');
    });

    it('should reject if request not found', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue(null);

      const result = await service.acceptRematch('request-1', 'user-2', createGameFn);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Rematch request not found');
    });

    it('should reject if request not pending', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue({
        ...mockRequest,
        status: 'declined',
      });

      const result = await service.acceptRematch('request-1', 'user-2', createGameFn);

      expect(result.success).toBe(false);
      expect(result.error).toContain('already declined');
    });

    it('should reject if request expired', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue({
        ...mockRequest,
        expiresAt: new Date(Date.now() - 1000),
      });
      mockPrisma.rematchRequest.update.mockResolvedValue({
        ...mockRequest,
        status: 'expired',
      });

      const result = await service.acceptRematch('request-1', 'user-2', createGameFn);

      expect(result.success).toBe(false);
      expect(result.error).toContain('expired');
    });

    it('should reject if accepter was not a player', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue(mockRequest);

      const result = await service.acceptRematch('request-1', 'user-3', createGameFn);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Only players');
    });

    it('should reject if accepter is the requester', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue(mockRequest);

      const result = await service.acceptRematch('request-1', 'user-1', createGameFn);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Cannot accept your own');
    });
  });

  describe('declineRematch', () => {
    it('should decline rematch successfully', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue(mockRequest);
      mockPrisma.rematchRequest.update.mockResolvedValue({
        ...mockRequest,
        status: 'declined',
      });

      const result = await service.declineRematch('request-1', 'user-2');

      expect(result.success).toBe(true);
      expect(result.request?.status).toBe('declined');
    });

    it('should reject if request not found', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue(null);

      const result = await service.declineRematch('request-1', 'user-2');

      expect(result.success).toBe(false);
      expect(result.error).toBe('Rematch request not found');
    });

    it('should reject if request not pending', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue({
        ...mockRequest,
        status: 'accepted',
      });

      const result = await service.declineRematch('request-1', 'user-2');

      expect(result.success).toBe(false);
      expect(result.error).toContain('already accepted');
    });

    it('should reject if decliner was not a player', async () => {
      mockPrisma.rematchRequest.findUnique.mockResolvedValue(mockRequest);

      const result = await service.declineRematch('request-1', 'user-3');

      expect(result.success).toBe(false);
      expect(result.error).toContain('Only players');
    });
  });

  describe('getPendingRequest', () => {
    it('should return pending request if exists', async () => {
      mockPrisma.rematchRequest.findFirst.mockResolvedValue(mockRequest);

      const result = await service.getPendingRequest('game-1');

      expect(result).not.toBeNull();
      expect(result?.gameId).toBe('game-1');
    });

    it('should return null if no pending request', async () => {
      mockPrisma.rematchRequest.findFirst.mockResolvedValue(null);

      const result = await service.getPendingRequest('game-1');

      expect(result).toBeNull();
    });
  });

  describe('expireOldRequests', () => {
    it('should expire old pending requests', async () => {
      mockPrisma.rematchRequest.updateMany.mockResolvedValue({ count: 5 });

      const result = await service.expireOldRequests();

      expect(result).toBe(5);
      expect(mockPrisma.rematchRequest.updateMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            status: 'pending',
          }),
          data: expect.objectContaining({
            status: 'expired',
          }),
        })
      );
    });

    it('should return 0 when no requests to expire', async () => {
      mockPrisma.rematchRequest.updateMany.mockResolvedValue({ count: 0 });

      const result = await service.expireOldRequests();

      expect(result).toBe(0);
    });
  });

  describe('getRematchService singleton', () => {
    it('should return a RematchService instance', () => {
      const instance = getRematchService();
      expect(instance).toBeInstanceOf(RematchService);
    });

    it('should return same instance on multiple calls', () => {
      const instance1 = getRematchService();
      const instance2 = getRematchService();
      expect(instance1).toBe(instance2);
    });
  });
});

describe('RematchService error handling', () => {
  let service: RematchService;

  beforeEach(() => {
    jest.clearAllMocks();
    service = new RematchService();
  });

  it('should handle database errors in createRematchRequest', async () => {
    mockPrisma.rematchRequest.findFirst.mockRejectedValue(new Error('DB error'));

    const result = await service.createRematchRequest('game-1', 'user-1');

    expect(result.success).toBe(false);
    expect(result.error).toBe('Failed to create rematch request');
  });

  it('should handle database errors in acceptRematch', async () => {
    mockPrisma.rematchRequest.findUnique.mockRejectedValue(new Error('DB error'));

    const result = await service.acceptRematch('request-1', 'user-2', jest.fn());

    expect(result.success).toBe(false);
    expect(result.error).toBe('Failed to accept rematch');
  });

  it('should handle database errors in declineRematch', async () => {
    mockPrisma.rematchRequest.findUnique.mockRejectedValue(new Error('DB error'));

    const result = await service.declineRematch('request-1', 'user-2');

    expect(result.success).toBe(false);
    expect(result.error).toBe('Failed to decline rematch');
  });

  it('should throw on database errors in getPendingRequest', async () => {
    mockPrisma.rematchRequest.findFirst.mockRejectedValue(new Error('DB error'));

    await expect(service.getPendingRequest('game-1')).rejects.toThrow('DB error');
  });

  it('should throw on database errors in expireOldRequests', async () => {
    mockPrisma.rematchRequest.updateMany.mockRejectedValue(new Error('DB error'));

    await expect(service.expireOldRequests()).rejects.toThrow('DB error');
  });
});
