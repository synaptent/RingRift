/**
 * Unit tests for useBackendChat hook
 *
 * Tests cover:
 * - Initial state values
 * - Chat input state management
 * - Form submission handling
 * - Direct message sending
 * - Empty message filtering
 *
 * @jest-environment jsdom
 */

import { renderHook, act } from '@testing-library/react';
import { useBackendChat } from '../../src/client/hooks/useBackendChat';
import type { ChatMessage } from '../../src/client/hooks/useBackendChat';

describe('useBackendChat', () => {
  // Helper to create mock messages
  const createMessages = (count: number): ChatMessage[] =>
    Array.from({ length: count }, (_, i) => ({
      sender: `Player${i + 1}`,
      text: `Message ${i + 1}`,
    }));

  describe('Initial state', () => {
    it('should initialize with empty chat input', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      expect(result.current.chatInput).toBe('');
    });

    it('should return provided messages array', () => {
      const messages = createMessages(3);
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat(messages, sendChatMessage));

      expect(result.current.messages).toBe(messages);
      expect(result.current.messages).toHaveLength(3);
    });

    it('should handle empty messages array', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      expect(result.current.messages).toEqual([]);
    });
  });

  describe('Chat input management', () => {
    it('should update chat input value', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      act(() => {
        result.current.setChatInput('Hello world');
      });

      expect(result.current.chatInput).toBe('Hello world');
    });

    it('should clear chat input', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      act(() => {
        result.current.setChatInput('Some text');
      });
      expect(result.current.chatInput).toBe('Some text');

      act(() => {
        result.current.setChatInput('');
      });

      expect(result.current.chatInput).toBe('');
    });
  });

  describe('Form submission', () => {
    it('should send message on form submit', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      act(() => {
        result.current.setChatInput('Hello');
      });

      const mockEvent = {
        preventDefault: jest.fn(),
      } as unknown as React.FormEvent;

      act(() => {
        result.current.handleSubmit(mockEvent);
      });

      expect(mockEvent.preventDefault).toHaveBeenCalled();
      expect(sendChatMessage).toHaveBeenCalledWith('Hello');
    });

    it('should clear input after form submit', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      act(() => {
        result.current.setChatInput('Hello');
      });

      const mockEvent = {
        preventDefault: jest.fn(),
      } as unknown as React.FormEvent;

      act(() => {
        result.current.handleSubmit(mockEvent);
      });

      expect(result.current.chatInput).toBe('');
    });

    it('should not send empty message on form submit', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      const mockEvent = {
        preventDefault: jest.fn(),
      } as unknown as React.FormEvent;

      act(() => {
        result.current.handleSubmit(mockEvent);
      });

      expect(mockEvent.preventDefault).toHaveBeenCalled();
      expect(sendChatMessage).not.toHaveBeenCalled();
    });

    it('should not send whitespace-only message on form submit', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      act(() => {
        result.current.setChatInput('   ');
      });

      const mockEvent = {
        preventDefault: jest.fn(),
      } as unknown as React.FormEvent;

      act(() => {
        result.current.handleSubmit(mockEvent);
      });

      expect(sendChatMessage).not.toHaveBeenCalled();
    });
  });

  describe('Direct message sending', () => {
    it('should send message directly via sendMessage', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      act(() => {
        result.current.sendMessage('Direct message');
      });

      expect(sendChatMessage).toHaveBeenCalledWith('Direct message');
    });

    it('should not send empty message via sendMessage', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      act(() => {
        result.current.sendMessage('');
      });

      expect(sendChatMessage).not.toHaveBeenCalled();
    });

    it('should not send whitespace-only message via sendMessage', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      act(() => {
        result.current.sendMessage('   ');
      });

      expect(sendChatMessage).not.toHaveBeenCalled();
    });
  });

  describe('Messages prop updates', () => {
    it('should reflect updated messages array', () => {
      const sendChatMessage = jest.fn();
      const initialMessages = createMessages(2);

      const { result, rerender } = renderHook(
        ({ messages }) => useBackendChat(messages, sendChatMessage),
        { initialProps: { messages: initialMessages } }
      );

      expect(result.current.messages).toHaveLength(2);

      const updatedMessages = createMessages(5);
      rerender({ messages: updatedMessages });

      expect(result.current.messages).toHaveLength(5);
    });

    it('should maintain chat input across message updates', () => {
      const sendChatMessage = jest.fn();
      const initialMessages = createMessages(2);

      const { result, rerender } = renderHook(
        ({ messages }) => useBackendChat(messages, sendChatMessage),
        { initialProps: { messages: initialMessages } }
      );

      act(() => {
        result.current.setChatInput('Typing...');
      });

      const updatedMessages = createMessages(5);
      rerender({ messages: updatedMessages });

      expect(result.current.chatInput).toBe('Typing...');
    });
  });

  describe('Multiple submissions', () => {
    it('should handle multiple form submissions', () => {
      const sendChatMessage = jest.fn();
      const { result } = renderHook(() => useBackendChat([], sendChatMessage));

      const mockEvent = {
        preventDefault: jest.fn(),
      } as unknown as React.FormEvent;

      // First submission
      act(() => {
        result.current.setChatInput('First');
      });
      act(() => {
        result.current.handleSubmit(mockEvent);
      });

      // Second submission
      act(() => {
        result.current.setChatInput('Second');
      });
      act(() => {
        result.current.handleSubmit(mockEvent);
      });

      expect(sendChatMessage).toHaveBeenCalledTimes(2);
      expect(sendChatMessage).toHaveBeenNthCalledWith(1, 'First');
      expect(sendChatMessage).toHaveBeenNthCalledWith(2, 'Second');
    });
  });
});
