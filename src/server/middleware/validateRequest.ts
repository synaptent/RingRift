import { Request, Response, NextFunction } from 'express';
import { ZodSchema } from 'zod';

const validatedQueryByRequest = new WeakMap<Request, unknown>();

/**
 * Middleware factory that validates `req.body` against a Zod schema.
 *
 * On success the parsed (and potentially transformed) data replaces
 * `req.body`, giving downstream handlers a typed, validated payload.
 *
 * On failure the ZodError is forwarded to the centralized error handler
 * which already knows how to map it to a 400 response with field-level
 * details.
 *
 * @example
 * ```ts
 * router.post('/register', validateBody(RegisterSchema), asyncHandler(async (req, res) => {
 *   // req.body is now the parsed RegisterSchema output
 * }));
 * ```
 */
export function validateBody<T>(schema: ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction): void => {
    try {
      req.body = schema.parse(req.body);
      next();
    } catch (err) {
      next(err);
    }
  };
}

/**
 * Middleware factory that validates `req.query` against a Zod schema.
 *
 * Express 5 exposes `req.query` through a getter, so replacing it throws at
 * runtime. Parsed output (including coerced values and defaults) is stored in
 * request-scoped middleware state and can be read with `getValidatedQuery`.
 *
 * @example
 * ```ts
 * router.get('/games', validateQuery(GameListingQuerySchema), asyncHandler(async (req, res) => {
 *   const { status, limit, offset } = getValidatedQuery<GameListingQueryInput>(req);
 * }));
 * ```
 */
export function validateQuery<T>(schema: ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction): void => {
    try {
      validatedQueryByRequest.set(req, schema.parse(req.query));
      next();
    } catch (err) {
      next(err);
    }
  };
}

/**
 * Return the parsed query produced by `validateQuery` for this request.
 *
 * Route handlers should call this only after the matching validation
 * middleware. Failing closed here prevents a handler from accidentally using
 * unvalidated query input if the middleware is removed or reordered.
 */
export function getValidatedQuery<T>(req: Request): T {
  if (!validatedQueryByRequest.has(req)) {
    throw new Error('Validated query is unavailable; ensure validateQuery runs first');
  }

  return validatedQueryByRequest.get(req) as T;
}

/**
 * Middleware factory that validates `req.params` against a Zod schema.
 *
 * @example
 * ```ts
 * router.get('/:gameId', validateParams(GameIdParamSchema), asyncHandler(async (req, res) => {
 *   const { gameId } = req.params;
 * }));
 * ```
 */
export function validateParams<T>(schema: ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction): void => {
    try {
      req.params = schema.parse(req.params) as typeof req.params;
      next();
    } catch (err) {
      next(err);
    }
  };
}
