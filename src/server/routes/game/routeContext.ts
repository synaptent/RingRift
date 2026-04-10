import type { WebSocketServer } from '../../websocket/server';

export interface GameRouteContext {
  getWebSocketServer(): WebSocketServer | null;
}
