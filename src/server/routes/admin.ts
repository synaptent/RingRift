import { Router } from 'express';
import type { AuthenticatedRequest } from '../middleware/auth';
import { authenticate, authorize } from '../middleware/auth';
import { orchestratorRollout } from '../services/OrchestratorRolloutService';
import { shadowComparator } from '../services/ShadowModeComparator';
import { config } from '../config';

const router = Router();

/**
 * @openapi
 * /admin/orchestrator/status:
 *   get:
 *     summary: Get orchestrator rollout and shadow-mode status
 *     description: |
 *       Returns the current orchestrator rollout configuration, circuit breaker
 *       state, error rates, and high-level shadow-mode comparison metrics.
 *       This endpoint is restricted to admin users.
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Orchestrator status snapshot
 */
router.get(
  '/orchestrator/status',
  authenticate,
  authorize(['admin']),
  (req: AuthenticatedRequest, res) => {
    const orchestratorConfig = config.featureFlags.orchestrator;
    const cbState = orchestratorRollout.getCircuitBreakerState();
    const errorRatePercent = orchestratorRollout.getErrorRate();
    const shadowMetrics = shadowComparator.getMetrics();

    res.json({
      success: true,
      data: {
        config: {
          adapterEnabled: orchestratorConfig.adapterEnabled,
          rolloutPercentage: orchestratorConfig.rolloutPercentage,
          shadowModeEnabled: orchestratorConfig.shadowModeEnabled,
          allowlistUsers: orchestratorConfig.allowlistUsers,
          denylistUsers: orchestratorConfig.denylistUsers,
          circuitBreaker: {
            enabled: orchestratorConfig.circuitBreaker.enabled,
            errorThresholdPercent: orchestratorConfig.circuitBreaker.errorThresholdPercent,
            errorWindowSeconds: orchestratorConfig.circuitBreaker.errorWindowSeconds,
          },
        },
        circuitBreaker: {
          isOpen: cbState.isOpen,
          errorCount: cbState.errorCount,
          requestCount: cbState.requestCount,
          windowStart: new Date(cbState.windowStart).toISOString(),
          errorRatePercent,
        },
        shadow: shadowMetrics,
      },
    });
  }
);

export default router;
