/**
 * Tests for #83 and #85: circuit breaker half-open concurrency and
 * single-source-of-truth state machine.
 *
 * Before #83's fix, a circuit breaker in the "open" state that had passed
 * its cooldown window would admit EVERY concurrent request that happened
 * to arrive in the half-open window. That violated the "one trial request
 * only" contract and meant a still-broken upstream saw the same thundering
 * herd the breaker was supposed to prevent.
 *
 * The test below constructs that scenario and asserts only one call
 * reaches the underlying function during half-open, while all others
 * short-circuit with "Circuit breaker is open" until the trial resolves.
 */

import { CircuitBreaker } from '../../src/server/services/AIServiceClient';

type Breaker = InstanceType<typeof CircuitBreaker>;
const getBreaker = (): Breaker => new CircuitBreaker();

async function openTheBreaker(breaker: ReturnType<typeof getBreaker>) {
  // Trip 5 failures in a row to cross the threshold.
  const boom = async () => {
    throw new Error('forced failure');
  };
  for (let i = 0; i < 5; i++) {
    await expect(breaker.execute(boom)).rejects.toThrow(/forced failure/);
  }
  expect(breaker.getStatus().state).toBe('open');
}

describe('CircuitBreaker half-open single-trial gate (#83, #85)', () => {
  // Jest fake timers override Date.now globally so the breaker sees our
  // controlled clock when it calls `Date.now()` internally.  Only "modern"
  // fake timers expose setSystemTime; we keep "doNotFake" empty so the
  // breaker's own Promise microtasks still run on the real event loop.
  beforeAll(() => {
    jest.useFakeTimers({ doNotFake: ['setImmediate', 'queueMicrotask'] });
  });

  afterAll(() => {
    jest.useRealTimers();
  });

  beforeEach(() => {
    jest.setSystemTime(new Date(1_000_000));
  });

  const advanceTime = (ms: number) => {
    jest.setSystemTime(new Date(Date.now() + ms));
  };

  it('state machine exposes "closed" / "open" / "half_open" via getStatus', async () => {
    const breaker = getBreaker();
    const status = breaker.getStatus();
    expect(['closed', 'open', 'half_open']).toContain(status.state);
    // #85: isOpen is derived from state, so the two fields must agree.
    expect(status.isOpen).toBe(status.state === 'open');
  });

  it('exactly one concurrent request reaches fn() during half-open (#83)', async () => {
    const breaker = getBreaker();
    await openTheBreaker(breaker);

    // Cross the cooldown threshold.
    advanceTime(61000);

    // Fire 5 concurrent calls.  A slow fn() holds the trial open so other
    // callers see the half-open + trial-in-flight state.
    let resolveTrial: (value: string) => void = () => {};
    const trialPromise = new Promise<string>((res) => {
      resolveTrial = res;
    });
    const slowFn = jest.fn(() => trialPromise);

    // Fire all 5 synchronously. Each async function will run up to its
    // first `await`; the first call reaches `await fn()` (fn is called
    // synchronously before the await yields), the remaining 4 throw in
    // the else-if branch before reaching fn.
    const settled: Array<'pending' | 'fulfilled' | 'rejected'> = [
      'pending',
      'pending',
      'pending',
      'pending',
      'pending',
    ];
    const errors: Array<Error | null> = [null, null, null, null, null];
    const promises = Array.from({ length: 5 }, (_, i) =>
      breaker.execute(slowFn).then(
        () => {
          settled[i] = 'fulfilled';
        },
        (err) => {
          settled[i] = 'rejected';
          errors[i] = err;
        }
      )
    );

    // Yield one microtask cycle so the 4 synchronous rejections settle.
    await Promise.resolve();
    await Promise.resolve();

    // Exactly ONE call reached fn.
    expect(slowFn).toHaveBeenCalledTimes(1);

    // 4 rejections with the "breaker is open" error.
    const rejectionCount = settled.filter((s) => s === 'rejected').length;
    expect(rejectionCount).toBe(4);
    const rejectionErrors = errors.filter((e): e is Error => e !== null);
    for (const err of rejectionErrors) {
      expect(err.message).toMatch(/Circuit breaker is open/);
    }

    // Release the trial so Jest cleans up the pending promise.
    resolveTrial('ok');
    await Promise.all(promises);
    expect(settled.filter((s) => s === 'fulfilled')).toHaveLength(1);
    expect(breaker.getStatus().state).toBe('closed');
  });

  it('after half-open trial succeeds, breaker returns to closed', async () => {
    const breaker = getBreaker();
    await openTheBreaker(breaker);
    advanceTime(61000);

    const ok = async () => 'healthy';
    await breaker.execute(ok);
    expect(breaker.getStatus().state).toBe('closed');
    expect(breaker.getStatus().isOpen).toBe(false);
  });

  it('after half-open trial fails, breaker re-opens', async () => {
    const breaker = getBreaker();
    await openTheBreaker(breaker);
    advanceTime(61000);

    const stillBroken = async () => {
      throw new Error('still broken');
    };
    await expect(breaker.execute(stillBroken)).rejects.toThrow(/still broken/);

    // One failure during half-open does not by itself re-trip the 5-strike
    // threshold counter, but because the trial failed, subsequent calls
    // again face the cooldown check.  Concretely: the state will have
    // transitioned to open only if failureCount >= threshold.  Here the
    // counter persists from the initial 5 strikes, so state stays 'open'.
    expect(breaker.getStatus().state).toBe('open');
  });

  it('isCircuitOpen() reports true during half-open with trial in flight', async () => {
    const breaker = getBreaker();
    await openTheBreaker(breaker);
    advanceTime(61000);

    let resolveTrial: (v: string) => void = () => {};
    const trialPromise = new Promise<string>((res) => {
      resolveTrial = res;
    });

    const inflight = breaker.execute(() => trialPromise);
    // Trial is in flight — the public "is it safe to call" check must
    // return true so callers know to use fallback.
    expect(breaker.isCircuitOpen()).toBe(true);

    resolveTrial('ok');
    await inflight;
    expect(breaker.isCircuitOpen()).toBe(false);
  });
});
