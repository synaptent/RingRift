import express, { type Request } from 'express';
import request from 'supertest';
import { z } from 'zod';
import { getValidatedQuery, validateQuery } from '../../src/server/middleware/validateRequest';

describe('validateQuery middleware', () => {
  const QuerySchema = z.object({
    limit: z.coerce.number().int().min(1).default(20),
    enabled: z.enum(['true', 'false']).transform((value) => value === 'true'),
  });

  it('exposes parsed query values without replacing the Express 5 query getter', async () => {
    const app = express();
    app.get('/query', validateQuery(QuerySchema), (req, res) => {
      res.json({
        raw: req.query,
        parsed: getValidatedQuery<z.infer<typeof QuerySchema>>(req),
      });
    });

    const response = await request(app).get('/query?limit=7&enabled=true').expect(200);

    expect(response.body).toEqual({
      raw: { limit: '7', enabled: 'true' },
      parsed: { limit: 7, enabled: true },
    });
  });

  it('fails closed when a handler reads validated query state without middleware', () => {
    expect(() => getValidatedQuery({} as Request)).toThrow(
      'Validated query is unavailable; ensure validateQuery runs first'
    );
  });
});
